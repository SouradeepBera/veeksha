"""Prefetch worker for session generation and scheduling."""

import time
from typing import List, Optional

import atomics

from veeksha.core.context import WorkerContext
from veeksha.core.session import Session
from veeksha.generator.session.base import BaseSessionGenerator
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

logger = init_logger(__name__)


class SharedSessionCounter:
    """Lock-free shared counter that hands out session slots across workers.
    """

    def __init__(self, max_sessions: int = -1):
        self.max_sessions = max_sessions
        self._count = atomics.atomic(width=8, atype=atomics.UINT)

    def claim(self) -> Optional[int]:
        slot = self._count.fetch_inc(atomics.MemoryOrder.RELAXED)
        if 0 <= self.max_sessions <= slot:
            return None
        return slot

    @property
    def count(self) -> int:
        claimed = self._count.load(atomics.MemoryOrder.RELAXED)
        if self.max_sessions < 0:
            return claimed
        return min(claimed, self.max_sessions)


class PrefetchWorker:
    """Worker that generates sessions and schedules them with the traffic scheduler.

    This worker pulls sessions from the session generator and feeds them to the
    traffic scheduler, which then manages the dispatch timing of individual requests.
    """

    # unthrottled for the first _BURST_DURATION_S seconds, then throttles
    _BURST_DURATION_S = 5.0
    _MAX_POLL_INTERVAL_S = 0.05

    def __init__(
        self,
        traffic_scheduler: BaseTrafficScheduler,
        session_generator: BaseSessionGenerator,
        worker_context: WorkerContext,
        session_counter: SharedSessionCounter,
        pregenerated_sessions: Optional[List[Session]] = None,
    ):
        """Initialize the prefetch worker.

        The session generator and traffic scheduler are expected to be thread-safe

        Args:
            traffic_scheduler: Scheduler to schedule sessions with
            session_generator: Generator to get sessions from
            worker_context: Worker context with stop event
            session_counter: Shared counter for tracking sessions across workers
            pregenerated_sessions: Optional list of pre-generated sessions to use
        """
        self.traffic_scheduler = traffic_scheduler
        self.session_generator = session_generator
        self.worker_context = worker_context
        self.session_counter = session_counter
        self._pregenerated_sessions = pregenerated_sessions

    def _get_poll_interval(self) -> float:
        """Calculate poll interval based on runtime duration.

        Unthrottled for the first _BURST_DURATION_S seconds, then throttles
        to _MAX_POLL_INTERVAL_S.

        Returns:
            Poll interval in seconds.
        """
        if time.monotonic() - self._start_time < self._BURST_DURATION_S:
            return 0.0
        return self._MAX_POLL_INTERVAL_S

    def _generate_session(self) -> Optional[Session]:
        """Generate next session in a thread-safe manner."""
        slot = self.session_counter.claim()
        if slot is None:
            return None  # exhausted

        # If we have pre-generated sessions, use those
        if self._pregenerated_sessions is not None:
            if slot >= len(self._pregenerated_sessions):
                return None
            return self._pregenerated_sessions[slot]

        # Otherwise generate on-the-fly
        try:
            return self.session_generator.generate_session()
        except StopIteration:
            logger.debug(
                "Prefetch worker %s: generator exhausted",
                self.worker_context.worker_id,
            )
            return None

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Prefetch worker %s starting", self.worker_context.worker_id)

        self._start_time = time.monotonic()

        while not self.worker_context.stop_event.is_set():
            session = self._generate_session()
            if session is None:
                logger.info(
                    "Prefetch worker %s: no more sessions to generate",
                    self.worker_context.worker_id,
                )
                break

            # Schedule the session with traffic scheduler
            self.traffic_scheduler.schedule_session(session)

            current_session_count = self.session_counter.count
            if current_session_count % 100 == 0:
                logger.debug(
                    "Prefetch progress: %d sessions generated",
                    current_session_count,
                )

            # Throttle (burst at start, then steady-state)
            time.sleep(self._get_poll_interval())

        logger.debug("Prefetch worker %s exiting", self.worker_context.worker_id)
