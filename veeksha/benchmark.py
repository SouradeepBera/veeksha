import os
import sys
import sysconfig
import threading
import time
from dataclasses import replace
from queue import Queue
from typing import Optional, Set

from veeksha.benchmark_utils import (
    _init_output_dir,
    _monitor_for_completion,
    build_evaluator,
    maybe_run_warmup,
)
from veeksha.client.registry import ClientRegistry
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.endpoint import EndpointConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.thread_pool import ThreadPoolManager
from veeksha.core.trace_recorder import TraceRecorder
from veeksha.evaluator.base import EvaluationResult
from veeksha.generator.session.registry import SessionGeneratorRegistry
from veeksha.health import HealthChecker, maybe_build_tts_zombie_probe
from veeksha.logger import init_logger
from veeksha.orchestration.benchmark_orchestrator import managed_server
from veeksha.traffic.registry import TrafficSchedulerRegistry
from veeksha.wandb_integration import (
    maybe_finish_wandb_run,
    maybe_init_wandb_run,
    maybe_log_benchmark_artifacts,
    maybe_log_benchmark_scalars,
)
from veeksha.workers import CompletionWorker, DispatchWorker, PrefetchWorker
from veeksha.workers.client_runner import ClientRunnerManager
from veeksha.workers.prefetch import SharedSessionCounter

logger = init_logger(__name__)


def _warn_if_gil_enabled(stage: str) -> None:
    """Warn when the GIL is active on a free-threaded build.

    A C extension that does not declare free-threading support re-enables the
    GIL at import time unless the process runs with ``-Xgil=0`` /
    ``PYTHON_GIL=0``. This can happen mid-run (e.g. the first
    ``librosa.load`` lazily imports ``msgpack``), silently serializing every
    client worker thread and invalidating high-concurrency measurements.
    """
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return
    if sys._is_gil_enabled():
        logger.warning(
            "The GIL is enabled at %s on a free-threaded Python build; "
            "client worker threads serialize on it. Launch with -Xgil=0 or "
            "PYTHON_GIL=0 to keep it disabled.",
            stage,
        )


def _maybe_pregenerate_sessions(benchmark_config, session_generator) -> Optional[list]:
    """Pre-generate sessions when enabled in runtime config."""
    if not (
        benchmark_config.runtime.pregenerate_sessions
        and benchmark_config.runtime.max_sessions > 0
    ):
        return None

    logger.info("Pre-generating %d sessions...", benchmark_config.runtime.max_sessions)
    pregenerated_sessions = []
    for _ in range(benchmark_config.runtime.max_sessions):
        try:
            session = session_generator.generate_session()
            pregenerated_sessions.append(session)
        except StopIteration:
            logger.warning(
                "Session generator exhausted at %d sessions",
                len(pregenerated_sessions),
            )
            break
    logger.info(
        "Pre-generation complete: %d sessions ready", len(pregenerated_sessions)
    )
    return pregenerated_sessions


def _run_main_loop(
    session_generator,
    traffic_scheduler,
    evaluator,
    client,
    runtime_config,
    trace_recorder=None,
    benchmark_start_time: Optional[float] = None,
    pregenerated_sessions: Optional[list] = None,
) -> None:
    """Run the main benchmark loop with all workers."""
    logger.info("Starting main loop")
    _warn_if_gil_enabled("benchmark start")
    if benchmark_start_time is None:
        benchmark_start_time = time.monotonic()

    num_client_threads = runtime_config.num_client_threads
    if num_client_threads is None:
        # Provision client workers for the offered load (the sweep planner
        # already does this; direct configs get the same protection): an
        # under-provisioned pool serializes per-session sends and shows up
        # as phantom server-side latency at high concurrency.
        target_sessions = getattr(traffic_scheduler, "target_concurrent_sessions", None)
        num_client_threads = (
            max(3, -(-int(target_sessions) // 8)) if target_sessions else 3
        )
    client_queues = [Queue() for _ in range(num_client_threads)]
    output_queue = Queue()
    stop_event = threading.Event()
    generator_lock = threading.Lock()

    session_counter = SharedSessionCounter(max_sessions=runtime_config.max_sessions)

    client_runner = ClientRunnerManager(
        client=client,
        input_queues=client_queues,
        output_queue=output_queue,
        traffic_scheduler=traffic_scheduler,
    )

    pool_manager = ThreadPoolManager(stop_event=stop_event)

    pool_manager.create_pool(
        name="prefetch",
        worker_class=PrefetchWorker,
        worker_kwargs={
            "traffic_scheduler": traffic_scheduler,
            "session_generator": session_generator,
            "generator_lock": generator_lock,
            "session_counter": session_counter,
            "pregenerated_sessions": pregenerated_sessions,
        },
        pool_size=1,
    )

    pool_manager.create_pool(
        name="dispatch",
        worker_class=DispatchWorker,
        worker_kwargs={
            "traffic_scheduler": traffic_scheduler,
            "client_queues": client_queues,
            "evaluator": evaluator,
            "trace_recorder": trace_recorder,
        },
        pool_size=runtime_config.num_dispatcher_threads,
    )

    pool_manager.create_pool(
        name="completion",
        worker_class=CompletionWorker,
        worker_kwargs={
            "output_queue": output_queue,
            "traffic_scheduler": traffic_scheduler,
            "evaluator": evaluator,
        },
        pool_size=runtime_config.num_completion_threads,
    )

    if trace_recorder:
        trace_recorder.start()

    client_runner.start()
    pool_manager.start_all()

    logger.info(
        f"Started {pool_manager.get_total_thread_count()} worker threads "
        f"and {client_runner.get_worker_count()} client workers"
    )

    benchmark_start = benchmark_start_time
    benchmark_timeout = runtime_config.benchmark_timeout
    timeout_triggered = False
    pre_timeout_request_ids: Set[str] = set()

    try:
        pending_in_flight = _monitor_for_completion(
            traffic_scheduler,
            evaluator,
            pool_manager,
            benchmark_start,
            benchmark_timeout,
            timeout_triggered,
            pre_timeout_request_ids,
            max_sessions=runtime_config.max_sessions,
            post_timeout_grace_seconds=runtime_config.post_timeout_grace_seconds,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted, stopping")
        pending_in_flight = set()

    # The GIL can flip on mid-run via lazy extension imports; re-check so a
    # serialized run is at least loudly reported.
    _warn_if_gil_enabled("benchmark end")

    stop_event.set()
    pool_manager.join_pool("prefetch", timeout=1.0)
    pool_manager.join_pool("dispatch", timeout=1.0)

    if trace_recorder:
        trace_recorder.stop()

    logger.info("Stopping client runner...")
    client_runner.stop()
    if not pending_in_flight:
        client_runner.wait()

    # immediate=False (unlike the client queues): results already queued are
    # measurements, so completion workers must drain them before ShutDown is
    # raised.  immediate=True would discard the backlog.
    output_queue.shutdown(immediate=False)
    pool_manager.join_pool("completion", timeout=1.0)


def _run_benchmark(
    benchmark_config: BenchmarkConfig,
):
    """Run the benchmark and return evaluation results.

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        EvaluationResult from the evaluator.
    """

    seed_manager = SeedManager(benchmark_config.seed)

    # get session generator
    tokenizer_provider = benchmark_config.client.build_tokenizer_provider()

    append_min_tokens_instruction = False
    if (
        hasattr(benchmark_config.client, "use_min_tokens_prompt_fallback")
        and benchmark_config.client.use_min_tokens_prompt_fallback  # type: ignore
    ):
        append_min_tokens_instruction = True
        logger.info(
            "Min tokens prompt fallback enabled in config. "
            "Will append instructions to prompts for minimum token control."
        )

    session_generator_kwargs = {
        "config": benchmark_config.session_generator,
        "seed_manager": seed_manager,
        "tokenizer_provider": tokenizer_provider,
    }

    # lm-eval uses runtime.max_sessions as the only sample-size knob.
    if (
        benchmark_config.session_generator.get_type()
        == SessionGeneratorRegistry.get_key_from_str("lmeval")
    ):
        session_generator_kwargs["max_sessions"] = benchmark_config.runtime.max_sessions

    if (
        benchmark_config.session_generator.get_type()
        == SessionGeneratorRegistry.get_key_from_str("synthetic")
    ):
        session_generator_kwargs["append_min_tokens_instruction"] = (
            append_min_tokens_instruction
        )

    session_generator = SessionGeneratorRegistry.get(
        benchmark_config.session_generator.get_type(),
        **session_generator_kwargs,
    )

    # get traffic scheduler, client
    traffic_scheduler = TrafficSchedulerRegistry.get(
        benchmark_config.traffic_scheduler.get_type(),
        config=benchmark_config.traffic_scheduler,
        seed_manager=seed_manager,
    )

    client = ClientRegistry.get(
        benchmark_config.client.get_type(),
        config=benchmark_config.client,
        tokenizer_provider=tokenizer_provider,
    )

    # some session generators might define a warmup phase
    maybe_run_warmup(session_generator, client)

    # Pre-generate all sessions if requested (before starting timer)
    pregenerated_sessions = _maybe_pregenerate_sessions(
        benchmark_config, session_generator
    )

    # Snapshot server-side finished-session counters (Vajra TTS endpoints only)
    # so the post-run health check can detect zombie sessions.
    tts_zombie_probe = maybe_build_tts_zombie_probe(benchmark_config)
    if tts_zombie_probe is not None:
        tts_zombie_probe.capture_start()

    benchmark_start_time = time.monotonic()
    traffic_scheduler.reset_reference_time()

    # get evaluator
    evaluator = build_evaluator(
        benchmark_config,
        seed_manager=seed_manager,
        session_generator=session_generator,
        benchmark_start_time=benchmark_start_time,
    )

    # trace recorder
    trace_recorder = None
    if benchmark_config.trace_recorder.enabled:
        # ensure output dirs exists for traces
        os.makedirs(f"{benchmark_config.output_dir}/traces", exist_ok=True)
        trace_recorder = TraceRecorder(
            f"{benchmark_config.output_dir}/traces",
            benchmark_start_time,
            benchmark_config.trace_recorder,
        )
        trace_recorder.start()

    os.makedirs(f"{benchmark_config.output_dir}/metrics", exist_ok=True)

    try:
        _run_main_loop(
            session_generator=session_generator,
            traffic_scheduler=traffic_scheduler,
            evaluator=evaluator,
            client=client,
            runtime_config=benchmark_config.runtime,
            trace_recorder=trace_recorder,
            benchmark_start_time=benchmark_start_time,
            pregenerated_sessions=pregenerated_sessions,
        )
    finally:
        if trace_recorder:
            trace_recorder.stop()

    if tts_zombie_probe is not None:
        tts_zombie_probe.capture_end()

    logger.info("Finalizing evaluator...")
    # finalize and save results
    finalize_started_at = time.monotonic()
    result = evaluator.finalize()
    logger.info(
        "Benchmark phase 'evaluator_finalize' took %.2fs",
        time.monotonic() - finalize_started_at,
    )

    save_started_at = time.monotonic()
    evaluator.save(f"{benchmark_config.output_dir}/metrics")
    logger.info(
        "Benchmark phase 'evaluator_save' took %.2fs",
        time.monotonic() - save_started_at,
    )

    # health checks
    logger.info("Running health checks...")
    health_started_at = time.monotonic()
    health_checker = HealthChecker(
        trace_file=f"{benchmark_config.output_dir}/traces/dispatch_trace.jsonl",
        metrics_file=f"{benchmark_config.output_dir}/metrics/request_level_metrics.jsonl",
        benchmark_config=benchmark_config,
        tts_zombie_probe=tts_zombie_probe,
    )
    health_checker.run_and_save(
        f"{benchmark_config.output_dir}/health_check_results.txt"
    )
    logger.info(
        "Benchmark phase 'health_checks' took %.2fs",
        time.monotonic() - health_started_at,
    )

    return result


def _with_endpoint(
    benchmark_config: BenchmarkConfig, endpoint: EndpointConfig
) -> BenchmarkConfig:
    return replace(
        benchmark_config,
        client=endpoint.apply_to_client_config(benchmark_config.client),
        endpoint=endpoint,
        server=None,
    )


def _run_initialized_benchmark(
    benchmark_config: BenchmarkConfig,
) -> EvaluationResult:
    maybe_init_wandb_run(benchmark_config, run_kind="benchmark")
    try:
        result = _run_benchmark(benchmark_config)
        maybe_log_benchmark_scalars(benchmark_config.output_dir)
        maybe_log_benchmark_artifacts(benchmark_config)
        return result
    finally:
        maybe_finish_wandb_run(benchmark_config.output_dir)


def run_benchmark_with_endpoint(
    benchmark_config: BenchmarkConfig,
    endpoint: EndpointConfig,
) -> EvaluationResult:
    """Run one benchmark against an endpoint managed by the caller."""
    logger.info("Running benchmark with config:\n%s", benchmark_config)
    _init_output_dir(benchmark_config)
    return _run_initialized_benchmark(_with_endpoint(benchmark_config, endpoint))


def manage_benchmark_run(
    benchmark_config: BenchmarkConfig,
) -> EvaluationResult:
    """Run a benchmark, handling optional server orchestration.

    If a server config exists, launch it and apply its endpoint. Otherwise,
    apply an explicit endpoint when configured and run directly.

    Args:
        benchmark_config: The benchmark configuration.

    Returns:
        Evaluation result from the configured evaluators.
    """
    logger.info("Running benchmark with config:\n%s", benchmark_config)
    _init_output_dir(benchmark_config)

    if benchmark_config.server is not None:
        logger.info("Launching %s server...", benchmark_config.server.engine)
        with managed_server(
            benchmark_config.server,
            output_dir=benchmark_config.output_dir,
        ) as server_info:
            endpoint = server_info["endpoint"]
            logger.info("Server ready at %s", endpoint.api_base)
            try:
                return _run_initialized_benchmark(
                    _with_endpoint(benchmark_config, endpoint)
                )
            finally:
                logger.info("Server shutting down...")

    if benchmark_config.endpoint is not None:
        benchmark_config = _with_endpoint(
            benchmark_config,
            benchmark_config.endpoint,
        )
    return _run_initialized_benchmark(benchmark_config)


if __name__ == "__main__":
    from veeksha.cli.benchmarks import main

    main()
