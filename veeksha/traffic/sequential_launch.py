"""Sequential-launch traffic scheduler.

Activates all sessions upfront and assigns incrementing dispatch tickets.
Dispatch ordering is enforced at the client level via a shared
:class:`DispatchTracker`: each request blocks until its ticket is called,
and the counter advances once the HTTP request is sent.
"""

from __future__ import annotations

import heapq
import threading
import time
from typing import Mapping, Optional

from veeksha.config.traffic import SequentialLaunchTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import children, parents, ready_at
from veeksha.traffic.base import BaseTrafficScheduler
from veeksha.traffic.dispatch_tracker import DispatchTracker
from veeksha.traffic.session_state import ScheduledItem, ScheduledSessionState
from veeksha.types import ChannelModality


class SequentialLaunchTrafficScheduler(BaseTrafficScheduler):
    """Scheduler that dispatches sessions one at a time via ticket ordering.

    All sessions are activated immediately when scheduled.  Each root request
    receives an incrementing ``dispatch_ticket``.  The shared
    :class:`DispatchTracker` ensures that client workers send HTTP requests
    strictly in ticket order, with near-zero inter-dispatch gap.
    """

    def __init__(
        self, config: SequentialLaunchTrafficConfig, seed_manager: SeedManager
    ):
        super().__init__(config, seed_manager)
        self._condition = threading.Condition()
        self._start_monotonic = time.monotonic()
        self._ready_queue: list[ScheduledItem] = []
        self._sessions: dict[int, ScheduledSessionState] = {}
        self._request_to_session: dict[int, tuple[int, int]] = {}
        self._dispatch_tracker = DispatchTracker(ordering=config.ordering)
        self._next_ticket: int = 0

    @property
    def dispatch_tracker(self) -> Optional[DispatchTracker]:
        return self._dispatch_tracker

    def _now(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _add_to_ready_queue(self, ready_at_time: float, request: Request) -> None:
        heapq.heappush(
            self._ready_queue,
            ScheduledItem(
                ready_at=ready_at_time, request_id=request.id, request=request
            ),
        )
        self._condition.notify_all()

    def _activate_session_locked(self, session: Session) -> None:
        start_time = self._now()
        state = ScheduledSessionState(
            session=session,
            session_start_time=start_time,
            completions={},
            pending_nodes=set(session.session_graph.nodes.keys()),
            queued_nodes=set(),
            cancel_on_failure=self.config.cancel_session_on_failure,
        )
        self._sessions[session.id] = state

        graph = session.session_graph
        for node_id in list(state.pending_nodes):
            if not parents(graph, node_id):
                node_ready_at = start_time + graph.nodes[node_id].wait_after_ready
                request = session.requests[node_id]
                request.dispatch_ticket = self._next_ticket
                self._next_ticket += 1
                self._add_to_ready_queue(node_ready_at, request)
                self._request_to_session[request.id] = (session.id, node_id)
                state.pending_nodes.discard(node_id)
                state.queued_nodes.add(node_id)

    def schedule_session(self, session: Session) -> None:
        with self._condition:
            self._activate_session_locked(session)

    def wait_for_ready(
        self, timeout: float = 0.001
    ) -> Optional[tuple[Request, int, int, float]]:
        with self._condition:
            result = self._try_pop_ready_locked()
            if result is not None:
                return result

            if self._ready_queue:
                wait_time = min(timeout, self._ready_queue[0].ready_at - self._now())
                wait_time = max(0.0001, wait_time)
            else:
                wait_time = timeout

            self._condition.wait(timeout=wait_time)
            return self._try_pop_ready_locked()

    def _try_pop_ready_locked(self) -> Optional[tuple[Request, int, int, float]]:
        if not self._ready_queue:
            return None

        if self._ready_queue[0].ready_at <= self._now():
            item = heapq.heappop(self._ready_queue)
            request = item.request

            session_id, node_id = self._request_to_session[request.id]
            state = self._sessions[session_id]
            session_size = len(state.session.requests)

            self._populate_history(request, state, node_id)

            # ready_at (offset from scheduler start) as an absolute monotonic time.
            scheduler_ready_at = item.ready_at + self._start_monotonic
            return (request, session_id, session_size, scheduler_ready_at)
        return None

    def pop_ready(self) -> Optional[tuple[Request, int, int, float]]:
        with self._condition:
            return self._try_pop_ready_locked()

    def _populate_history(
        self, request: Request, state: ScheduledSessionState, node_id: int
    ) -> None:
        graph = state.session.session_graph
        incoming_edges = parents(graph, node_id)
        history_parents = [e for e in incoming_edges if e.is_history_parent]

        if len(history_parents) > 1:
            raise ValueError(f"Ambiguous history inheritance for node {node_id}")

        if history_parents:
            parent_id = history_parents[0].src
            request.history = state.node_histories.get(parent_id, [])
        else:
            request.history = []

    def notify_completion(
        self,
        request_id: int,
        completed_at_monotonic: float,
        success: bool,
        channel_responses: Optional[Mapping[ChannelModality, ChannelResponse]] = None,
    ) -> None:
        with self._condition:
            session_id, node_id = self._request_to_session.pop(request_id)
            state = self._sessions[session_id]
            completed_at = completed_at_monotonic - self._start_monotonic

            state.completions[node_id] = completed_at
            state.queued_nodes.discard(node_id)

            self._record_history(state, node_id, success, channel_responses)

            if not success and state.cancel_on_failure:
                state.is_canceled = True
                state.pending_nodes.clear()
                if not state.queued_nodes:
                    del self._sessions[session_id]
                return

            # Release children that are now ready.
            graph = state.session.session_graph
            for edge in children(graph, node_id):
                child_id = edge.dst
                if child_id not in state.pending_nodes:
                    continue
                node_ready_at = ready_at(graph, child_id, state.completions)
                if node_ready_at is not None:
                    request = state.session.requests[child_id]
                    self._add_to_ready_queue(node_ready_at, request)
                    self._request_to_session[request.id] = (session_id, child_id)
                    state.pending_nodes.discard(child_id)
                    state.queued_nodes.add(child_id)

            if not state.pending_nodes and not state.queued_nodes:
                del self._sessions[session_id]

    def has_pending_work(self) -> bool:
        with self._condition:
            return bool(self._sessions or self._ready_queue)

    def get_in_flight_request_ids(self) -> set[int]:
        with self._condition:
            return set(self._request_to_session.keys())

    def _record_history(
        self,
        state: ScheduledSessionState,
        node_id: int,
        success: bool,
        channel_responses: Optional[Mapping[ChannelModality, ChannelResponse]],
    ) -> None:
        graph = state.session.session_graph
        outgoing = children(graph, node_id)
        if any(edge.is_history_parent for edge in outgoing):
            request = state.session.requests[node_id]
            current_node_history = list(request.history)

            content_blocks = []
            if ChannelModality.TEXT in request.channels:
                text_content = request.channels[ChannelModality.TEXT]
                content_blocks.append(
                    {"type": "text", "text": text_content.input_text}  # type: ignore
                )
            if ChannelModality.IMAGE in request.channels:
                image_content = request.channels[ChannelModality.IMAGE]
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_content.input_image},  # type: ignore
                    }
                )
            if ChannelModality.AUDIO in request.channels:
                audio_content = request.channels[ChannelModality.AUDIO]
                content_blocks.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_content.input_audio},  # type: ignore
                    }
                )
            if ChannelModality.VIDEO in request.channels:
                video_content = request.channels[ChannelModality.VIDEO]
                content_blocks.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": video_content.input_video},  # type: ignore
                    }
                )

            if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
                current_node_history.append(
                    {"role": "user", "content": content_blocks[0]["text"]}
                )
            elif content_blocks:
                current_node_history.append({"role": "user", "content": content_blocks})

            if success and channel_responses is not None:
                if ChannelModality.TEXT in channel_responses:
                    response_text = channel_responses[ChannelModality.TEXT].content
                    current_node_history.append(
                        {"role": "assistant", "content": response_text}
                    )

            state.node_histories[node_id] = current_node_history
