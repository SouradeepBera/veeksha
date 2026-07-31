from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Mapping, Optional, Set, Tuple

from veeksha.config.traffic import BaseTrafficConfig
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.traffic.dispatch_tracker import DispatchTracker


class BaseTrafficScheduler:
    def __init__(self, config: BaseTrafficConfig, seed_manager: SeedManager):
        self.config = config
        self.seed_manager = seed_manager

    @abstractmethod
    def schedule_session(self, session: Session) -> None:
        """Schedule a session for dispatch."""
        raise NotImplementedError

    @abstractmethod
    def pop_ready(self) -> Optional[Tuple[Request, int, int, float]]:
        """Pop a ready request.

        Returns (request, session_id, session_size, scheduler_ready_at), or None.
        ``scheduler_ready_at`` is the scheduled ready instant in the
        ``time.monotonic()`` domain.
        """
        raise NotImplementedError

    @abstractmethod
    def wait_for_ready(
        self, timeout: float = 0.001
    ) -> Optional[Tuple[Request, int, int, float]]:
        """Wait up to ``timeout`` seconds for a ready request.

        Same return shape as :meth:`pop_ready`; None on timeout.
        """
        raise NotImplementedError

    @abstractmethod
    def notify_completion(
        self,
        request_id: int,
        completed_at_monotonic: float,
        success: bool,
        channel_responses: Optional[Mapping[ChannelModality, ChannelResponse]] = None,
    ) -> None:
        """Notify the scheduler that a request has completed."""
        raise NotImplementedError

    @abstractmethod
    def has_pending_work(self) -> bool:
        """Check if there are pending sessions or in-flight requests."""
        raise NotImplementedError

    @abstractmethod
    def get_in_flight_request_ids(self) -> Set[int]:
        """Return the set of request IDs currently in-flight."""
        raise NotImplementedError

    @property
    def dispatch_tracker(self) -> Optional[DispatchTracker]:
        """Optional dispatch tracker for ticket-based ordering."""
        return None

    def notify_request_sent(self, request_id: int) -> None:
        """Called when the server acknowledges a request (HTTP 200 received).

        The default implementation is a no-op.  Subclasses (e.g.
        :class:`SequentialLaunchTrafficScheduler`) may use this to gate
        activation of pending sessions.
        """
        return

    def reset_reference_time(self) -> None:
        """Optional hook invoked before the benchmark starts dispatching."""
        return
