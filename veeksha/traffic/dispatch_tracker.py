"""Thread-safe dispatch ordering via ticket counter."""

import threading


class DispatchTracker:
    """Gates request dispatch using a ticket counter.

    Each request is assigned an incrementing ticket number.  A request may
    dispatch when ``ticket <= counter``.  After the HTTP request is sent,
    the counter advances so the next ticket unblocks.
    """

    def __init__(self, ordering: str = "dispatch") -> None:
        self._counter = 0
        self._condition = threading.Condition()
        self._ordering = ordering

    @property
    def ordering(self) -> str:
        return self._ordering

    def wait_for_turn(self, ticket: int) -> None:
        """Block until ``ticket <= counter``."""
        with self._condition:
            while ticket > self._counter:
                self._condition.wait()

    def advance(self, ticket: int) -> None:
        """Advance counter and wake waiters."""
        with self._condition:
            self._counter = max(self._counter, ticket + 1)
            self._condition.notify_all()
