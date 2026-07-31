"""Concurrent traffic scheduler maintaining a target number of active sessions."""

import heapq
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Set, Tuple

from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import children, parents, ready_at
from veeksha.traffic.base import BaseTrafficScheduler
from veeksha.traffic.session_state import ScheduledItem, ScheduledSessionState
from veeksha.types import ChannelModality


class ConcurrentTrafficScheduler(BaseTrafficScheduler):
    """Scheduler maintaining a target number of concurrent active sessions.

    Sessions are queued and activated immediately when capacity is available.
    Time base: seconds since scheduler creation (monotonic reference).
    """

    def __init__(self, config: ConcurrentTrafficConfig, seed_manager: SeedManager):
        super().__init__(config, seed_manager)
        self._target_concurrent = config.target_concurrent_sessions
        self._rampup_seconds = config.rampup_seconds
        self._rampup_complete = False
        self._condition = threading.Condition()
        self._start_monotonic = time.monotonic()
        self._ready_queue: List[ScheduledItem] = []
        self._sessions: Dict[int, ScheduledSessionState] = {}
        self._session_locks: Dict[int, threading.Lock] = {}
        self._request_to_session: Dict[int, Tuple[int, int]] = {}
        self._pending_sessions: Deque[Session] = deque()

    def _now(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _current_target_concurrency(self) -> int:
        if self._rampup_complete:
            return self._target_concurrent
        if self._now() >= self._rampup_seconds:
            self._rampup_complete = True
            return self._target_concurrent
        return int(self._target_concurrent * (self._now() / self._rampup_seconds))

    def _add_to_ready_queue(self, ready_at_time: float, request: Request) -> None:
        """Add item to ready queue and signal waiting dispatchers."""
        heapq.heappush(
            self._ready_queue,
            ScheduledItem(
                ready_at=ready_at_time, request_id=request.id, request=request
            ),
        )
        self._condition.notify()

    def _active_session_count(self) -> int:
        return len(self._sessions)

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
        self._session_locks[session.id] = threading.Lock()

        # queue root nodes. The session lock is deliberately not taken here:
        # both the state and its lock are published inside this critical
        # section, so no other thread can obtain a reference to either until
        # _condition is released. Taking it would nest _condition -> session,
        # inverting the ordering used everywhere else.
        graph = session.session_graph
        for node_id in list(state.pending_nodes):
            if not parents(graph, node_id):
                node_ready_at = start_time + graph.nodes[node_id].wait_after_ready
                request = session.requests[node_id]
                self._add_to_ready_queue(node_ready_at, request)
                self._request_to_session[request.id] = (session.id, node_id)
                state.pending_nodes.discard(node_id)
                state.queued_nodes.add(node_id)

    def _try_activate_pending_locked(self) -> None:
        while (
            self._pending_sessions
            and self._active_session_count() < self._current_target_concurrency()
        ):
            session = self._pending_sessions.popleft()
            self._activate_session_locked(session)

    def schedule_session(self, session: Session) -> None:
        with self._condition:
            self._pending_sessions.append(session)
            self._try_activate_pending_locked()

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
        result = self._try_pop()
        if result is not None:
            return result

        with self._condition:
            if self._ready_queue:
                wait_time = min(timeout, self._ready_queue[0].ready_at - self._now())
                wait_time = max(0.0001, wait_time)
            else:
                wait_time = timeout

            self._condition.wait(timeout=wait_time)

        # check again after waking
        return self._try_pop()

    def _try_pop(self) -> Optional[Tuple[Request, int, int, float]]:
        """Try to pop a ready item. Must be called holding no lock.

        Takes _condition and the session lock in turn, never together.
        """
        with self._condition:
            if not self._ready_queue:
                return None

            if self._ready_queue[0].ready_at > self._now():
                return None

            item = heapq.heappop(self._ready_queue)
            request = item.request

            # A queued node stays in state.queued_nodes until it completes, and
            # teardown requires that set to be empty, so both lookups are live.
            session_id, node_id = self._request_to_session[request.id]
            state = self._sessions[session_id]
            session_lock = self._session_locks[session_id]

        with session_lock:
            session_size = len(state.session.requests)
            self._populate_history(request, state, node_id)

        # ready_at (offset from scheduler start) as an absolute monotonic time.
        scheduler_ready_at = item.ready_at + self._start_monotonic
        return (request, session_id, session_size, scheduler_ready_at)

    def pop_ready(self) -> Optional[Tuple[Request, int, int, float]]:
        return self._try_pop()

    def _populate_history(
        self, request: Request, state: ScheduledSessionState, node_id: int
    ) -> None:
        """Populate request history from session state based on parent edges."""
        graph = state.session.session_graph
        incoming_edges = parents(graph, node_id)
        history_parents = [e for e in incoming_edges if e.is_history_parent]

        # identify history parent
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
            session_lock = self._session_locks[session_id]

        completed_at = completed_at_monotonic - self._start_monotonic

        # The session lock decides; _condition only applies the decision. The
        # two are never held at the same time, so there is no lock ordering to
        # get wrong and no path from one session's lock to another's.
        released: List[Tuple[float, Request, int]] = []
        with session_lock:
            state.completions[node_id] = completed_at
            state.queued_nodes.discard(node_id)

            # record history if needed (only if I am a history parent)
            self._record_history(state, node_id, success, channel_responses)

            # cancel session on failure
            if not success and state.cancel_on_failure:
                state.is_canceled = True
                state.pending_nodes.clear()
            else:
                # children might be ready; claim the ones this thread releases
                graph = state.session.session_graph
                for edge in children(graph, node_id):
                    child_id = edge.dst
                    if child_id not in state.pending_nodes:
                        continue
                    node_ready_at = ready_at(graph, child_id, state.completions)
                    if node_ready_at is None:
                        continue
                    # Claiming here is what makes a multi-parent child safe: two
                    # parents completing concurrently both pass the membership
                    # test only if the discard is not serialized with it.
                    # queued_nodes is added to before pending_nodes is discarded
                    # so the child is never absent from both sets.
                    state.queued_nodes.add(child_id)
                    state.pending_nodes.discard(child_id)
                    released.append(
                        (node_ready_at, state.session.requests[child_id], child_id)
                    )

            # Each thread removes its own node from queued_nodes under this
            # lock, so exactly one can observe the transition to empty.
            session_finished = not state.pending_nodes and not state.queued_nodes

        if released:
            with self._condition:
                for node_ready_at, request, child_id in released:
                    self._add_to_ready_queue(node_ready_at, request)
                    self._request_to_session[request.id] = (session_id, child_id)

        if session_finished:
            with self._condition:
                del self._sessions[session_id]
                del self._session_locks[session_id]
                self._try_activate_pending_locked()

    def has_pending_work(self) -> bool:
        """Check if there are pending sessions or in-flight requests."""
        with self._condition:
            return bool(self._pending_sessions or self._sessions or self._ready_queue)

    def get_in_flight_request_ids(self) -> Set[int]:
        """Return the set of request IDs currently in-flight."""
        with self._condition:
            return set(self._request_to_session.keys())

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
