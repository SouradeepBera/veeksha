"""Completion worker for processing completed requests."""

import time
from queue import Empty, Queue

from veeksha.core.context import WorkerContext
from veeksha.core.response import RequestResult
from veeksha.evaluator.base import BaseEvaluator
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler
from veeksha.traffic.load_distributor import LoadDistributor

logger = init_logger(__name__)


QUEUE_GET_TIMEOUT_S = 0.1
DRAIN_MAX_EMPTY_POLLS = 5


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
        load_distributor: LoadDistributor,
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
        self.load_distributor = load_distributor

    def _process_result(self, result: RequestResult) -> None:
        """Process a single request result."""
        result.result_processed_at = time.monotonic()

        self.load_distributor.notify_completion(result.client_worker_id)

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

        while not self.worker_context.stop_event.is_set():
            try:
                item = self.output_queue.get(timeout=QUEUE_GET_TIMEOUT_S)
            except Empty:
                continue

            # sentinel
            if item is None:
                break

            self._process_result(item)

        self._drain()

        logger.debug("Completion worker %s exiting", self.worker_context.worker_id)

    def _drain(self) -> None:
        """Drain any remaining results from the queue."""
        logger.debug(
            "Completion worker %s: draining queue", self.worker_context.worker_id
        )
        empty_polls = 0
        while empty_polls < DRAIN_MAX_EMPTY_POLLS:
            try:
                item = self.output_queue.get_nowait()
                # sentinel
                if item is None:
                    break
                self._process_result(item)
                empty_polls = 0
            except Empty:
                empty_polls += 1
                time.sleep(0.01)
