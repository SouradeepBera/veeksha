"""Completion worker for processing completed requests."""

import time
from queue import Queue, ShutDown

from veeksha.core.context import WorkerContext
from veeksha.core.response import RequestResult
from veeksha.evaluator.base import BaseEvaluator
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

logger = init_logger(__name__)


class CompletionWorker:
    """Worker that processes completed requests from client output queue.

    This worker:
    1. Receives RequestResult from client output queue
    2. Notifies traffic scheduler of completion
    3. Records completion with evaluator
    """

    def __init__(
        self,
        output_queue: Queue,
        traffic_scheduler: BaseTrafficScheduler,
        evaluator: BaseEvaluator,
        worker_context: WorkerContext,
    ):
        """Initialize the completion worker.

        Args:
            output_queue: Queue receiving RequestResult from client workers
            traffic_scheduler: Scheduler to notify of completions
            evaluator: Evaluator to record completions with
            worker_context: Worker context with stop event
        """
        self.output_queue = output_queue
        self.traffic_scheduler = traffic_scheduler
        self.evaluator = evaluator
        self.worker_context = worker_context

    def _process_result(self, result: RequestResult) -> None:
        """Process a single request result."""
        result.result_processed_at = time.monotonic()

        self.traffic_scheduler.notify_completion(
            request_id=result.request_id,
            completed_at_monotonic=result.client_completed_at,  # type: ignore
            success=result.success,
            channel_responses=result.channels if result.success else None,
        )

        error = Exception(result.error_msg) if result.error_msg else None
        self.evaluator.record_request_completed(
            request_id=result.request_id,
            session_id=result.session_id,
            completed_at=result.client_completed_at,  # type: ignore
            response=result,
            error=error,
        )

    def run(self) -> None:
        """Main worker loop."""
        logger.debug("Completion worker %s starting", self.worker_context.worker_id)

        while True:
            try:
                item = self.output_queue.get()
            except ShutDown:
                # The queue was closed with immediate=False, so ShutDown is
                # only raised once every queued result has been handed out.
                # Reaching here therefore means the backlog is fully drained;
                # no separate drain pass is needed.
                break

            self._process_result(item)

        logger.debug("Completion worker %s exiting", self.worker_context.worker_id)
