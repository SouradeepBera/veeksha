"""Response data structures for the new Veeksha framework.

These dataclasses bridge the gap between LLM client responses and what evaluators need.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from veeksha.types import ChannelModality


@dataclass
class ChannelResponse:
    """Response data for a single channel.

    Attributes:
        modality: The channel modality (TEXT, IMAGE, AUDIO, VIDEO)
        content: Modality-specific content (e.g., text string, image bytes)
        metrics: Channel-specific metrics (e.g., inter_token_times for TEXT)
    """

    modality: ChannelModality
    content: Any
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestResult:
    """Result from executing a single request.

    This is the output of an LLM client call, containing both timing information
    and per-channel response data that evaluators can consume.

    Attributes:
        request_id: Unique request identifier
        session_id: Session this request belongs to
        session_total_requests: Total number of requests in the session
        client_worker_id: Index of the client worker thread that ran this
            request; -1 until the worker stamps it
        channels: Per-channel response data
        success: True if request completed without error
        error_code: HTTP error code if request failed
        error_msg: Error message if request failed

        Lifecycle timestamps (all monotonic):
        - scheduler_ready_at: When scheduler marked request ready
        - scheduler_dispatched_at: When dispatcher put request in client queue
        - client_picked_up_at: When client worker dequeued request
        - client_completed_at: When HTTP response finished
        - result_processed_at: When completion worker finished processing
    """

    request_id: int
    session_id: int

    session_total_requests: int = 1

    # Stamped by ClientWorker._process_request() on the way out, alongside the
    # lifecycle timestamps -- clients do not know which worker is running them.
    client_worker_id: int = -1

    # per-channel responses
    channels: Dict[ChannelModality, ChannelResponse] = field(default_factory=dict)

    success: bool = True
    error_code: Optional[int] = None
    error_msg: Optional[str] = None

    # Lifecycle telemetry (monotonic)
    scheduler_ready_at: Optional[float] = (
        None  # DispatchWorker.run() after wait_for_ready
    )
    scheduler_dispatched_at: Optional[float] = (
        None  # DispatchWorker.run() after queue.put
    )
    client_picked_up_at: Optional[float] = (
        None  # ClientWorker._process_request() on dequeue
    )
    client_sent_at: Optional[float] = (
        None  # t_cs: request handed to the transport (set by wired clients)
    )
    client_completed_at: Optional[float] = (
        None  # LLM client send_request() after response
    )
    result_processed_at: Optional[float] = (
        None  # CompletionWorker._process_result() on entry
    )

    # Preflight timing telemetry (the client's own record book). Populated only
    # when the client is run with record_preflight_timing=True (the preflight
    # harness); left as None on normal benchmark runs so the production path pays
    # no cost. The scorer joins these with the server's record book by
    # request_id. All monotonic.
    chunk_recv_times: Optional[List[float]] = (
        None  # t_cr_i: client receipt of each response chunk
    )
    # Streaming-input clients only (realtime_tts / vajra / stt): per-input-segment
    # send times and the intended pacing deadlines, for input-delivery lag and
    # pacing-accuracy metrics. None for single-shot requests.
    input_send_times: Optional[List[float]] = None  # t_cs_i
    input_send_deadlines: Optional[List[float]] = None  # intended send instants
