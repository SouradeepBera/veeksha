from typing import Optional

from vidhi import field, frozen_dataclass


@frozen_dataclass
class RuntimeConfig:
    max_sessions: int = field(
        25, help="Maximum number of sessions to generate. -1 for unlimited."
    )
    benchmark_timeout: int = field(300, help="Benchmark timeout in seconds.")
    post_timeout_grace_seconds: int = field(
        -1,
        help="Grace period for in-flight requests after timeout. -1 waits for all, 0 exits immediately.",
    )
    num_dispatcher_threads: int = field(
        2, help="Number of threads for dispatching requests to workers."
    )
    num_completion_threads: int = field(
        8,
        help=(
            "Number of threads for processing completed requests. Completion "
            "workers also run per-request ASR scoring concurrently, so "
            "under-provisioning stretches the post-run drain."
        ),
    )
    num_client_threads: Optional[int] = field(
        8,
        help=(
            "Number of async worker threads for making concurrent requests. "
            "None provisions one thread per eight target-concurrent sessions, "
            "with a minimum of three. Under-provisioning stretches realtime "
            "send cadence and invalidates high-concurrency latency metrics."
        ),
    )
    pregenerate_sessions: bool = field(
        False,
        help="Pre-generate all sessions before starting benchmark timer. "
        "Requires max_sessions > 0.",
    )
