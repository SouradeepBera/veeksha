"""Rate-based traffic scheduler for dispatching sessions at a specified rate."""

import heapq
import threading
import time
from typing import Dict, List, Mapping, Optional, Set, Tuple

from veeksha.config.traffic import RateTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import children, parents, ready_at
from veeksha.generator.interval.registry import IntervalGeneratorRegistry
from veeksha.traffic.base import BaseTrafficScheduler
from veeksha.traffic.session_state import ScheduledItem, ScheduledSessionState
from veeksha.types import ChannelModality


class RateTrafficScheduler(BaseTrafficScheduler):
    """Scheduler for dispatching sessions at a specified rate.

    Time base: seconds since scheduler creation (monotonic reference).
    """

    def __init__(self, config: RateTrafficConfig, seed_manager: SeedManager):
        super().__init__(config, seed_manager)
        self._interval_gen = IntervalGeneratorRegistry.get(
            config.interval_generator.get_type(),
            config.interval_generator,
            rng=seed_manager.numpy_factory("interval")(),
        )
        self._condition = threading.Condition()
        self._next_start_time: float = 0.0
        self._start_monotonic = time.monotonic()
        self._ready_queue: List[ScheduledItem] = []
        self._sessions: Dict[int, ScheduledSessionState] = {}
        self._request_to_session: Dict[int, Tuple[int, int]] = {}

    def _now(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _add_to_ready_queue(self, ready_at_time: float, request: Request) -> None:
        """Add item to ready queue and signal waiting dispatchers."""
        heapq.heappush(
            self._ready_queue,
            ScheduledItem(
                ready_at=ready_at_time, request_id=request.id, request=request
            ),
        )
        self._condition.notify_all()

    def schedule_session(self, session: Session) -> None:
        with self._condition:
            start_time = self._next_start_time
            self._next_start_time += self._interval_gen.get_next_interval()

            state = ScheduledSessionState(
                session=session,
                session_start_time=start_time,
                completions={},
                pending_nodes=set(session.session_graph.nodes.keys()),
                queued_nodes=set(),
                cancel_on_failure=self.config.cancel_session_on_failure,
            )
            self._sessions[session.id] = state

            # queue root nodes
            graph = session.session_graph
            for node_id in list(state.pending_nodes):
                if not parents(graph, node_id):
                    node_ready_at = start_time + graph.nodes[node_id].wait_after_ready
                    request = session.requests[node_id]
                    self._add_to_ready_queue(node_ready_at, request)
                    self._request_to_session[request.id] = (session.id, node_id)
                    state.pending_nodes.discard(node_id)
                    state.queued_nodes.add(node_id)

    def wait_for_ready(
        self, timeout: float = 0.001
    ) -> Optional[Tuple[Request, int, int, float]]:
        """Wait for a ready request with timeout.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Tuple of (request, session_id, session_size, scheduler_ready_at) if
            ready, None if timeout.
        """
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

    def _try_pop_ready_locked(self) -> Optional[Tuple[Request, int, int, float]]:
        """Try to pop a ready item, must be called with lock held."""
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

    def pop_ready(self) -> Optional[Tuple[Request, int, int, float]]:
        with self._condition:
            return self._try_pop_ready_locked()

    def _populate_history(
        self, request: Request, state: ScheduledSessionState, node_id: int
    ) -> None:
        """Populate request history from session state based on parent edges."""
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

            # record history (only if I am a history parent)
            self._record_history(state, node_id, success, channel_responses)

            # cancel session on failure
            if not success and state.cancel_on_failure:
                state.is_canceled = True
                state.pending_nodes.clear()
                if not state.queued_nodes:
                    del self._sessions[session_id]
                return

            # children might be ready; release them
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

            # session is complete
            if not state.pending_nodes and not state.queued_nodes:
                del self._sessions[session_id]

    def has_pending_work(self) -> bool:
        """Check if there are pending sessions or in-flight requests."""
        with self._condition:
            return bool(self._sessions or self._ready_queue)

    def get_in_flight_request_ids(self) -> Set[int]:
        """Return the set of request IDs currently in-flight."""
        with self._condition:
            return set(self._request_to_session.keys())

    def reset_reference_time(self) -> None:
        """Align the scheduler's clock with the current monotonic time."""
        with self._condition:
            self._start_monotonic = time.monotonic()

    def _record_history(
        self,
        state: ScheduledSessionState,
        node_id: int,
        success: bool,
        channel_responses: Optional[Mapping[ChannelModality, ChannelResponse]],
    ) -> None:
        """Record history for a completed node if it's a history parent."""
        graph = state.session.session_graph
        outgoing = children(graph, node_id)
        if any(edge.is_history_parent for edge in outgoing):
            request = state.session.requests[node_id]

            current_node_history = list(request.history)  # Copy previous

            # user part
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

            # assistant part
            if success and channel_responses is not None:
                if ChannelModality.TEXT in channel_responses:
                    response_text = channel_responses[ChannelModality.TEXT].content
                    current_node_history.append(
                        {"role": "assistant", "content": response_text}
                    )

            state.node_histories[node_id] = current_node_history
