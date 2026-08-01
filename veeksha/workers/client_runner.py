"""Client runner manager for async request execution."""

import asyncio
import threading
import time
from queue import Queue, ShutDown
from typing import List, Optional

from veeksha.client.base import BaseLLMClient
from veeksha.core.response import RequestResult
from veeksha.logger import init_logger
from veeksha.traffic.base import BaseTrafficScheduler

logger = init_logger(__name__)


class ClientWorker:
    """Async client worker that processes requests from an input queue."""

    def __init__(
        self,
        worker_id: int,
        client: BaseLLMClient,
        input_queue: Queue,
        output_queue: Queue,
        traffic_scheduler: Optional[BaseTrafficScheduler] = None,
    ):
        self.worker_id = worker_id
        self.client = client
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.traffic_scheduler = traffic_scheduler

    def run(self) -> None:
        """Run the async event loop for this worker.

        Uses an explicit event loop instead of ``asyncio.run()`` to avoid the
        default teardown behavior: ``asyncio.run()`` calls
        ``loop.run_until_complete(gather(*all_tasks))`` which waits *forever*
        for in-flight tasks to finish.  At high concurrency with long output
        sequences the LLM server trickles tokens indefinitely, so those tasks
        never complete and the process hangs for hours.

        ``_run_async`` already cancels active tasks and gives them 2 s to
        respond.  Any that survive that window are destroyed when we close the
        loop — acceptable, since we explicitly told them to cancel.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async())
        finally:
            loop.close()

    async def _run_async(self) -> None:
        """Async main loop."""
        loop = asyncio.get_running_loop()
        active_tasks = set()

        while True:
            try:
                # avoid blocking the event loop
                item = await loop.run_in_executor(None, self.input_queue.get)
            except ShutDown:
                # ClientRunnerManager.stop() closed the queue
                break
            except Exception:
                continue

            task = asyncio.create_task(self._process_request(item))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        if active_tasks:
            logger.debug(
                "Client worker %d cancelling %d pending tasks",
                self.worker_id,
                len(active_tasks),
            )
            for task in active_tasks:
                task.cancel()
            await asyncio.wait(active_tasks, timeout=2.0)

        # Sessions bind to this loop, which ``run()`` closes next.
        await self.client.aclose()

        logger.debug("Client worker %d exiting", self.worker_id)

    async def _process_request(self, item) -> None:
        """Process a single request item."""
        (
            request,
            session_id,
            session_size,
            scheduler_ready_at,
            scheduler_dispatched_at,
        ) = item

        tracker = (
            self.traffic_scheduler.dispatch_tracker if self.traffic_scheduler else None
        )
        if tracker is not None:
            await tracker.wait_for_turn(request.dispatch_ticket)

        client_picked_up_at: float = time.monotonic()

        ordering = tracker.ordering if tracker is not None else "dispatch"

        def _on_request_sent() -> None:
            if self.traffic_scheduler is not None:
                self.traffic_scheduler.notify_request_sent(request.id)
            if ordering == "prefill" and tracker is not None:
                tracker.advance(request.dispatch_ticket)

        def _on_request_dispatched() -> None:
            if ordering == "dispatch" and tracker is not None:
                tracker.advance(request.dispatch_ticket)

        try:
            result = await self.client.send_request(
                request=request,
                session_id=session_id,
                session_total_requests=session_size,
                on_request_sent=_on_request_sent,
                on_request_dispatched=_on_request_dispatched,
            )
            if ordering == "request" and tracker is not None:
                tracker.advance(request.dispatch_ticket)
        except Exception as e:
            logger.exception(
                f"Client worker {self.worker_id}: Client raised unhandled exception"
            )
            # Ensure the tracker advances even on failure so subsequent
            # requests are not stuck waiting forever.
            if tracker is not None:
                tracker.advance(request.dispatch_ticket)

            result = RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_size,
                channels={},
                success=False,
                error_code=500,
                error_msg=f"Unhandled client exception: {str(e)}",
                client_completed_at=time.monotonic(),
            )

        # Set lifecycle timestamps
        result.scheduler_ready_at = scheduler_ready_at
        result.scheduler_dispatched_at = scheduler_dispatched_at
        result.client_picked_up_at = client_picked_up_at

        try:
            self.output_queue.put(result)
        except ShutDown:
            # Reachable only on the timeout path, where wait() is skipped so
            # teardown can close the output queue while this request is still
            # in flight.  The evaluator has already excluded these requests
            # via set_included_requests, so dropping the result is correct;
            # swallowing keeps it from surfacing as an unretrieved task
            # exception.
            logger.debug(
                "Client worker %d: output queue closed, dropping result for "
                "request %s",
                self.worker_id,
                request.id,
            )


class ClientRunnerManager:
    """Manager for a pool of async client worker threads."""

    def __init__(
        self,
        client: BaseLLMClient,
        input_queues: List[Queue],
        output_queue: Queue,
        traffic_scheduler: Optional[BaseTrafficScheduler] = None,
    ):
        """Initialize the client runner manager.

        Args:
            client: LLM client to use for requests
            input_queues: One input queue per worker
            output_queue: Shared output queue for results
            traffic_scheduler: Optional scheduler for request-sent notification
        """
        self.client = client
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.traffic_scheduler = traffic_scheduler
        self.workers: List[ClientWorker] = []
        self.threads: List[threading.Thread] = []

    def start(self) -> None:
        """Start all client worker threads."""
        for i, queue in enumerate(self.input_queues):
            worker = ClientWorker(
                worker_id=i,
                client=self.client,
                input_queue=queue,
                output_queue=self.output_queue,
                traffic_scheduler=self.traffic_scheduler,
            )
            self.workers.append(worker)

            thread = threading.Thread(
                target=worker.run,
                name=f"client-worker-{i}",
                daemon=True,
            )
            self.threads.append(thread)
            thread.start()

        logger.info("Started %d client worker threads", len(self.threads))

    def stop(self) -> None:
        """
        Close every input queue, waking all workers immediately.
        """
        for queue in self.input_queues:
            queue.shutdown(immediate=True)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for all worker threads to finish.

        Args:
            timeout: Maximum seconds to wait for all workers. ``None`` for
                indefinitely.

        Returns:
            True if every worker exited in time, False if the timeout expired.
        """
        start_time = time.monotonic()
        for thread in self.threads:
            join_timeout = None
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                remaining = max(0.0, timeout - elapsed)
                if remaining == 0.0 and thread.is_alive():
                    break
                join_timeout = remaining
            thread.join(join_timeout)

        alive_threads = [thread for thread in self.threads if thread.is_alive()]
        if alive_threads:
            logger.warning(
                "Timed out waiting for %d client worker threads", len(alive_threads)
            )
            return False

        logger.debug("All client worker threads joined")
        return True

    def get_worker_count(self) -> int:
        """Return number of workers."""
        return len(self.workers)
