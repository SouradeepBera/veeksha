"""Utilities used by the benchmark runner."""

import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Set, Tuple

import yaml
from tqdm import tqdm

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.utils import to_serializable_config_dict
from veeksha.core.seeding import SeedManager
from veeksha.evaluator.base import BaseEvaluator
from veeksha.evaluator.composite import CompositeEvaluator
from veeksha.evaluator.registry import EvaluatorRegistry
from veeksha.logger import init_logger
from veeksha.types import EvaluationType

logger = init_logger(__name__)

__all__ = [
    "_persist_config_yaml",
    "_init_output_dir",
    "build_evaluator",
    "maybe_run_warmup",
    "_monitor_for_completion",
]


def _persist_config_yaml(benchmark_config: BenchmarkConfig) -> str:
    """Write the resolved benchmark configuration to config.yml.

    Args:
        benchmark_config: The fully resolved benchmark configuration.

    Returns:
        Path to the persisted YAML file.
    """
    os.makedirs(benchmark_config.output_dir, exist_ok=True)
    config_dict = to_serializable_config_dict(benchmark_config)
    config_path = os.path.join(benchmark_config.output_dir, "config.yml")
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(
            config_dict,
            config_file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    logger.debug("Persisted benchmark config to %s", config_path)
    return config_path


def _init_output_dir(benchmark_config: BenchmarkConfig) -> str:
    """Resolve and prepare the final benchmark output directory.

    The function persists the config, computes its hash, and
    moves the config into a dated/hash-named subdirectory. The benchmark
    configuration's ``output_dir`` field is updated in-place to point to the
    resolved directory.

    Args:
        benchmark_config: Benchmark configuration to mutate.

    Returns:
        Path to the resolved output directory.
    """

    base_output_dir = benchmark_config.output_dir
    os.makedirs(base_output_dir, exist_ok=True)

    config_path = _persist_config_yaml(benchmark_config)
    with open(config_path, "rb") as config_file:
        config_bytes = config_file.read()
    config_hash = hashlib.sha1(config_bytes).hexdigest()[:8]

    timestamp_prefix = datetime.utcnow().strftime("%d_%m_%Y-%H_%M_%S")
    base_dir_name = f"{timestamp_prefix}-{config_hash}"
    resolved_output_dir = os.path.join(base_output_dir, base_dir_name)

    suffix = 1
    while os.path.exists(resolved_output_dir):
        suffix += 1
        resolved_output_dir = os.path.join(base_output_dir, f"{base_dir_name}-{suffix}")

    os.makedirs(resolved_output_dir, exist_ok=True)
    shutil.move(config_path, os.path.join(resolved_output_dir, "config.yml"))
    object.__setattr__(benchmark_config, "output_dir", resolved_output_dir)
    logger.info("Benchmark outputs will be stored in %s", resolved_output_dir)

    return resolved_output_dir


def maybe_run_warmup(session_generator, client) -> None:
    """Maybe run warmup sessions synchronously before benchmark.

    A warmup only runs the first request of each session specified.
    """

    import asyncio

    async def warmup_one(session):
        """Execute first request of a warmup session."""
        first_request = list(session.requests.values())[0]
        await client.send_request(first_request, session.id, 1)

    async def run_all(warmup_sessions):
        try:
            for session in tqdm(warmup_sessions, desc="Warmup", unit="sess"):
                await warmup_one(session)
        finally:
            # ``asyncio.run`` closes the loop this warmup's session is bound to;
            # the benchmark's client threads build their own afterwards.
            await client.aclose()

    if hasattr(session_generator, "get_warmup_sessions"):
        warmup_sessions = session_generator.get_warmup_sessions()
        if warmup_sessions:
            logger.info(f"Running warmup with {len(warmup_sessions)} sessions")
            asyncio.run(run_all(warmup_sessions))
            logger.info("Warmup completed")


def build_evaluator(
    benchmark_config: BenchmarkConfig,
    *,
    seed_manager: SeedManager,
    session_generator: Any,
    benchmark_start_time: float,
) -> BaseEvaluator:
    """Build an evaluator instance (or composite evaluator) for a benchmark run.

    Notes:
    - Performance evaluator(s) are ordered first so that `CompositeEvaluator` uses
      performance for progress/timeout behavior.
    - Accuracy evaluation requires access to the session generator to map
      request IDs back to lm-eval instances.

    Args:
        benchmark_config: Benchmark configuration.
        seed_manager: Seed manager for reproducibility.
        session_generator: Session generator used for this run.
        benchmark_start_time: Run start time (monotonic), passed to evaluators for
            time-normalization and artifact timestamps.

    Returns:
        A `BaseEvaluator` (single evaluator or `CompositeEvaluator`).
    """
    evaluator_configs = sorted(
        benchmark_config.evaluators,
        key=lambda cfg: 0 if cfg.get_type() == EvaluationType.PERFORMANCE else 1,
    )

    evaluator_instances: list[BaseEvaluator] = []
    for cfg in evaluator_configs:
        kwargs: Dict[str, Any] = {
            "config": cfg,
            "seed_manager": seed_manager,
            "output_dir": f"{benchmark_config.output_dir}/metrics",
            "benchmark_start_time": benchmark_start_time,
        }
        if cfg.get_type() == EvaluationType.PERFORMANCE:
            kwargs["client_type"] = benchmark_config.client.get_type()
        if cfg.get_type() == EvaluationType.ACCURACY_LMEVAL:
            kwargs["session_generator"] = session_generator
        evaluator_instances.append(EvaluatorRegistry.get(cfg.get_type(), **kwargs))

    if not evaluator_instances:
        raise ValueError("BenchmarkConfig.evaluators must be non-empty.")

    return (
        evaluator_instances[0]
        if len(evaluator_instances) == 1
        else CompositeEvaluator(evaluator_instances)
    )


def _init_pbar(max_sessions: int, benchmark_timeout: float) -> Tuple[Any, bool]:
    """Initialize progress bar based on benchmark mode."""
    if max_sessions > 0:
        pbar = tqdm(
            total=max_sessions,
            desc="Sessions",
            unit="sess",
            dynamic_ncols=True,
            bar_format="{desc}: {n}/{total} [{percentage:3.0f}%] | {rate_fmt} | Elapsed: {elapsed}",
        )
        return pbar, False

    pbar = tqdm(
        total=int(benchmark_timeout),
        desc="Benchmark",
        unit="s",
        dynamic_ncols=True,
        bar_format="{desc}: {elapsed}/{total} s [{percentage:3.0f}%] | Sessions: {postfix}",
    )
    pbar.set_postfix_str("0")
    return pbar, True


def _update_pbar(
    pbar,
    time_based_progress: bool,
    elapsed: float,
    total_done: int,
    state: Dict[str, int],
) -> None:
    """Update progress bar with current state."""
    if time_based_progress:
        elapsed_int = int(elapsed)
        if elapsed_int > state["last_time_update"]:
            pbar.update(elapsed_int - state["last_time_update"])
            state["last_time_update"] = elapsed_int
        if total_done > state["last_completed"]:
            pbar.set_postfix_str(str(total_done))
            state["last_completed"] = total_done
        return

    if total_done > state["last_completed"]:
        pbar.update(total_done - state["last_completed"])
        state["last_completed"] = total_done


def _progress_writer(max_sessions: int) -> Callable[[int], None]:
    """Return a callable that publishes benchmark progress as JSON.

    When ``VEEKSHA_PROGRESS_FILE`` is set (the launcher points the benchmark at
    a file there), the returned callable writes ``{"completed", "total"}`` to it
    atomically so consumers get structured progress without scraping the console.
    Returns a no-op when the variable is unset (standalone benchmark runs).
    """
    target = os.environ.get("VEEKSHA_PROGRESS_FILE")
    if not target:
        return lambda completed: None

    path = Path(target)
    tmp_path = path.with_name(path.name + ".tmp")
    total = max_sessions if max_sessions > 0 else None

    def write(completed: int) -> None:
        try:
            tmp_path.write_text(
                json.dumps({"completed": completed, "total": total}),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except OSError as exc:
            logger.debug("Failed to write progress file %s: %s", path, exc)

    return write


def _monitor_for_completion(
    traffic_scheduler,
    evaluator,
    pool_manager,
    benchmark_start: float,
    benchmark_timeout: float,
    timeout_triggered: bool,
    pre_timeout_request_ids: Set[str],
    max_sessions: int,
    post_timeout_grace_seconds: int = -1,
) -> Set[str]:
    """Observe worker progress and exit once requests settle.

    Returns:
        Set of request IDs that were still in-flight when monitoring stopped.
    """
    pbar, time_based_progress = _init_pbar(max_sessions, benchmark_timeout)
    pbar_state = {"last_completed": 0, "last_time_update": 0}
    timeout_start: float = 0.0
    in_flight_remaining: Set[str] = set()

    write_progress = _progress_writer(max_sessions)
    write_progress(0)
    last_written = 0

    try:
        while True:
            time.sleep(0.1)

            completed, errored, _ = evaluator.get_session_counts()
            total_done = completed + errored
            elapsed = time.monotonic() - benchmark_start

            _update_pbar(pbar, time_based_progress, elapsed, total_done, pbar_state)
            if total_done != last_written:
                write_progress(total_done)
                last_written = total_done

            if (
                not timeout_triggered
                and benchmark_timeout > 0
                and elapsed >= benchmark_timeout
            ):
                timeout_triggered = True
                timeout_start = time.monotonic()
                pre_timeout_request_ids = evaluator.get_registered_request_ids()
                in_flight = traffic_scheduler.get_in_flight_request_ids()
                pending = pre_timeout_request_ids & in_flight
                logger.info(
                    f"Benchmark timeout after {elapsed:.1f}s. "
                    f"Captured {len(pre_timeout_request_ids)} registered requests, "
                    f"{len(pending)} still in-flight."
                )

            prefetch_threads = pool_manager.thread_pools.get("prefetch", [])
            all_prefetch_done = all(not t.is_alive() for t in prefetch_threads)

            if timeout_triggered:
                current_in_flight = traffic_scheduler.get_in_flight_request_ids()
                remaining = pre_timeout_request_ids & current_in_flight

                # Check grace period if configured
                grace_elapsed = time.monotonic() - timeout_start
                if (
                    post_timeout_grace_seconds >= 0
                    and grace_elapsed >= post_timeout_grace_seconds
                ):
                    logger.warning(
                        f"Grace period of {post_timeout_grace_seconds}s expired. "
                        f"Force-exiting with {len(remaining)} requests still in-flight."
                    )
                    # Only include completed requests in metrics
                    completed_requests = pre_timeout_request_ids - remaining
                    evaluator.set_included_requests(completed_requests)
                    in_flight_remaining = remaining
                    break

                if not remaining:
                    logger.info("All pre-timeout requests completed")
                    evaluator.set_included_requests(pre_timeout_request_ids)
                    in_flight_remaining = set()
                    break
            elif all_prefetch_done and not traffic_scheduler.has_pending_work():
                logger.info("All sessions completed")
                in_flight_remaining = set()
                break
    finally:
        pbar.close()

    return in_flight_remaining
