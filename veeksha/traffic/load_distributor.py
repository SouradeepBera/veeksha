"""Credit-based load distribution across client worker queues."""

from collections import deque
import atomics

class LoadDistributor:
    """Routes dispatches to client queues on a completion-credit scheme.

    Every completed request mints one credit for the worker that ran it; each
    dispatch spends a credit, falling back to round-robin when none are
    available (cold start, or when dispatch outruns completion). This tracks
    real per-worker capacity instead of sampling queue depth, which is stale
    the moment it is read.

    Credits are held in a LIFO deque: the most recently freed worker is reused
    first, so a worker that just finished stays hot rather than the load being
    smeared across every thread.
    """

    def __init__(self, n_total_queues: int):
        self._credits: deque[int] = deque()
        self.n = n_total_queues
        self._rr = atomics.atomic(width=8, atype=atomics.UINT)

    def select_queue(self) -> int:
        """Return the index of the client queue to dispatch to."""
        try:
            return self._credits.popleft()
        except IndexError:
            return (self._rr.fetch_inc(atomics.MemoryOrder.RELAXED) % self.n)

    def notify_completion(self, worker_id: int) -> None:
        """Return a credit to the worker that finished a request."""
        if worker_id < 0:  # unstamped result; would index the last queue
            return
        self._credits.append(worker_id)
