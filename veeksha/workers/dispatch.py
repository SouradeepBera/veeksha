"""Dispatch worker for sending requests to client queues."""

import random
import time
from queue import Queue, ShutDown
from typing import TYPE_CHECKING, List, Optional

from veeksha.core.context import WorkerContext
from veeksha.evaluator.base import BaseEvaluator
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

if TYPE_CHECKING:
    from veeksha.core.trace_recorder import TraceRecorder

logger = init_logger(__name__)


class DispatchWorker:
    """Worker that polls ready requests from scheduler and dispatches to client queues.

    This worker:
    1. Waits for ready requests from the traffic scheduler (wait_for_ready)
    2. Registers requests with the evaluator
    3. Dispatches requests to client worker queues
    """

    _WAIT_TIMEOUT_S = 0.01

    def __init__(
        self,
        traffic_scheduler: BaseTrafficScheduler,
        client_queues: List[Queue],
        evaluator: BaseEvaluator,
        worker_context: WorkerContext,
        trace_recorder: Optional["TraceRecorder"] = None,
    ):
        """Initialize the dispatch worker.

        Args:
            traffic_scheduler: Scheduler to poll for ready requests
            client_queues: Queues to dispatch requests to (one per client worker)
            evaluator: Evaluator for registering request dispatch
            worker_context: Worker context with stop event
            trace_recorder: Optional recorder for dispatch traces
        """
        self.traffic_scheduler = traffic_scheduler
        self.client_queues = client_queues
        self.evaluator = evaluator
        self.worker_context = worker_context
        self.trace_recorder = trace_recorder

    def _select_queue(self) -> Queue:
        """Select a client queue using power-of-two load balancing"""
        n = len(self.client_queues)
        if n == 1:
            return self.client_queues[0]

        idx1, idx2 = random.sample(range(n), 2)

        q1 = self.client_queues[idx1]
        q2 = self.client_queues[idx2]

        return q1 if q1.qsize() <= q2.qsize() else q2

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Dispatch worker %s starting", self.worker_context.worker_id)

        while not self.worker_context.stop_event.is_set():
            result = self.traffic_scheduler.wait_for_ready(timeout=self._WAIT_TIMEOUT_S)

            if result is None:
                continue

            request, session_id, session_size, scheduler_ready_at = result
            # scheduler_ready_at: scheduled ready time; dispatched_at: actual
            # dequeue time.
            dispatched_at = time.monotonic()

            self.evaluator.register_request(
                request_id=request.id,
                session_id=session_id,
                dispatched_at=dispatched_at,
                channels=request.channels,
                requested_output=request.requested_output,
            )

            if self.trace_recorder:
                self.trace_recorder.record_dispatch(
                    request=request,
                    session_id=session_id,
                    session_size=session_size,
                    dispatched_at=dispatched_at,
                )

            queue = self._select_queue()
            try:
                queue.put(
                    (
                        request,
                        session_id,
                        session_size,
                        scheduler_ready_at,
                        dispatched_at,
                    )
                )
            except ShutDown:
                # Client queues were closed by ClientRunnerManager.stop();
                # there is nothing left to dispatch to, so skip the drain.
                logger.debug(
                    "Dispatch worker %s: client queues closed, exiting",
                    self.worker_context.worker_id,
                )
                return

        # Drain remaining ready requests
        self._drain()

        logger.debug("Dispatch worker %s exiting", self.worker_context.worker_id)

    def _drain(self) -> None:
        """Drain any remaining ready requests."""
        logger.debug(
            "Dispatch worker %s: draining scheduler", self.worker_context.worker_id
        )
        while True:
            result = self.traffic_scheduler.pop_ready()
            if result is None:
                break

            request, session_id, session_size, scheduler_ready_at = result
            dispatched_at = time.monotonic()

            self.evaluator.register_request(
                request_id=request.id,
                session_id=session_id,
                dispatched_at=dispatched_at,
                channels=request.channels,
                requested_output=request.requested_output,
            )

            if self.trace_recorder:
                self.trace_recorder.record_dispatch(
                    request=request,
                    session_id=session_id,
                    session_size=session_size,
                    dispatched_at=dispatched_at,
                )

            queue = self._select_queue()
            try:
                queue.put(
                    (
                        request,
                        session_id,
                        session_size,
                        scheduler_ready_at,
                        dispatched_at,
                    )
                )
            except ShutDown:
                logger.debug(
                    "Dispatch worker %s: client queues closed mid-drain",
                    self.worker_context.worker_id,
                )
                break
