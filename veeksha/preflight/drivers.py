"""Preflight drivers: run the REAL benchmark loop against a mock server.

The whole point of preflight is to measure the *actual* harness, so a driver
reuses veeksha's real components -- session generator, traffic scheduler,
dispatch/client/completion workers via ``benchmark._run_main_loop`` -- and only
substitutes (a) a deterministic mock server for the LLM endpoint and (b) a
capturing evaluator that keeps every ``RequestResult`` so the scorer can read
the paired client/server timestamps.

A word-split tokenizer is injected rather than the client's default HuggingFace
one, so preflight needs no ``transformers``/``tokenizers`` install and runs on a
bare free-threaded interpreter.

Each ``run_*_preflight`` takes its category workload config (input shape + mock
timing), the shared traffic scheduler, and the session count; it forces the
endpoint to the spawned mock and turns on ``record_preflight_timing``.
"""

from __future__ import annotations

from typing import List

from veeksha.benchmark import _run_main_loop
from veeksha.benchmark_utils import build_evaluator
from veeksha.client import ClientRegistry
from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.client import (
    BaseClientConfig,
    OpenAIChatCompletionsClientConfig,
    OpenAICompletionsClientConfig,
    RealtimeTTSClientConfig,
    STTClientConfig,
    TextPacingConfig,
    TTSClientConfig,
    VajraTTSStreamClientConfig,
)
from veeksha.config.evaluator import PerformanceEvaluatorConfig
from veeksha.config.generator.channel import (
    AudioChannelGeneratorConfig,
    TextChannelGeneratorConfig,
)
from veeksha.config.generator.length import FixedLengthGeneratorConfig
from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
from veeksha.config.generator.session_graph import (
    SingleRequestSessionGraphGeneratorConfig,
)
from veeksha.config.preflight import (
    PreflightSttCheckConfig,
    PreflightTextCheckConfig,
    PreflightTtsCheckConfig,
)
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.traffic import BaseTrafficConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.generator.session.registry import SessionGeneratorRegistry
from veeksha.logger import init_logger
from veeksha.preflight import scorer
from veeksha.preflight.models import ScoreReport
from veeksha.preflight.spawn import (
    MockServerHandle,
    spawn_mock_chat_server,
    spawn_mock_completions_server,
    spawn_mock_realtime_tts_server,
    spawn_mock_stt_server,
    spawn_mock_tts_server,
    spawn_mock_vajra_tts_server,
)
from veeksha.traffic.registry import TrafficSchedulerRegistry
from veeksha.types import ChannelModality

logger = init_logger(__name__)

# HTTP-TTS: the mock emits chunks of this size and the client reads with a
# matching chunk_size, so each server chunk is read 1:1 for the join.
_TTS_CHUNK_BYTES = 1024


def _preflight_encode(text: str):
    """Whitespace tokenizer that round-trips integer words stably.

    The prompt generator checks ``encode(decode(x)) == x`` on integer seeds, so a
    positional word-split encoder (which maps to range(n)) never stabilises. Here
    an integer word maps to itself and non-integer words (e.g. the mock's response
    text, which the client also encodes) map to a deterministic id.
    """
    ids = []
    for word in text.split():
        if word.lstrip("-").isdigit():
            ids.append(int(word))
        else:
            ids.append(sum(map(ord, word)))
    return ids


def _build_preflight_tokenizer_provider(model: str) -> TokenizerProvider:
    text_handle = TokenizerHandle(
        # Each token decodes to "<int> " -- the trailing space is a per-token
        # boundary so PromptStringGenerator (which tiles decoded tokens with
        # "".join) yields whitespace-separated words. Without it a prompt is one
        # space-less blob that segment_text treats as a single streamed delta.
        count_tokens=lambda text: len(text.split()),
        decode=lambda ids: "".join(f"{int(i)} " for i in ids),
        encode=_preflight_encode,
        get_vocab=lambda: list(range(10_000)),
    )
    # The audio channel generator ignores its tokenizer handle; a passthrough
    # satisfies the synthetic generator's per-channel lookup for AUDIO input.
    audio_handle = TokenizerHandle(
        count_tokens=lambda _: 0,
        decode=lambda ids: "",
        encode=lambda _: [],
    )
    return TokenizerProvider(
        {ChannelModality.TEXT: text_handle, ChannelModality.AUDIO: audio_handle},
        model_name=model,
    )


def _text_session_config(input_tokens: int) -> SyntheticSessionGeneratorConfig:
    return SyntheticSessionGeneratorConfig(
        session_graph=SingleRequestSessionGraphGeneratorConfig(),
        channels=[
            TextChannelGeneratorConfig(
                body_length_generator=FixedLengthGeneratorConfig(value=input_tokens)
            )
        ],
    )


def _audio_session_config(
    duration_seconds: float, sample_rate: int
) -> SyntheticSessionGeneratorConfig:
    return SyntheticSessionGeneratorConfig(
        session_graph=SingleRequestSessionGraphGeneratorConfig(),
        channels=[
            AudioChannelGeneratorConfig(
                duration_seconds=duration_seconds, sample_rate=sample_rate
            )
        ],
    )


def _stream_pacing(tokens_per_second: float, tokens_per_delta: int) -> TextPacingConfig:
    return TextPacingConfig(
        tokens_per_second=tokens_per_second,
        tokens_per_delta=tokens_per_delta,
        initial_delay_s=0.0,
    )


class CapturingEvaluator:
    """Wraps a real evaluator, keeping every completed ``RequestResult``.

    All calls the workers / monitor loop make delegate to the wrapped evaluator so
    termination behaviour is identical; we only tap ``record_request_completed``
    to stash the full result for scoring.
    """

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self.results: List[object] = []

    def record_request_completed(
        self, request_id, session_id, completed_at, response, error=None
    ) -> None:
        self.results.append(response)
        return self._wrapped.record_request_completed(
            request_id, session_id, completed_at, response, error
        )

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _build_benchmark_config(
    client_config: BaseClientConfig,
    session_config: SyntheticSessionGeneratorConfig,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        output_dir=output_dir,
        session_generator=session_config,
        traffic_scheduler=traffic_scheduler,
        client=client_config,
        runtime=runtime,
        server=None,
        # Preflight scores its own drift metrics; suppress the evaluator's dump.
        evaluators=[PerformanceEvaluatorConfig(stream_metrics=False)],
    )


def _run_capturing(benchmark_config: BenchmarkConfig) -> List[object]:
    """Run the real main loop for one BenchmarkConfig; return captured results.

    Mirrors ``benchmark._run_benchmark`` construction, but forces a word-split
    tokenizer and wraps the evaluator to capture results.
    """
    import time

    seed_manager = SeedManager(benchmark_config.seed)
    tokenizer_provider = _build_preflight_tokenizer_provider(
        benchmark_config.client.model
    )
    session_generator = SessionGeneratorRegistry.get(
        benchmark_config.session_generator.get_type(),
        config=benchmark_config.session_generator,
        seed_manager=seed_manager,
        tokenizer_provider=tokenizer_provider,
        append_min_tokens_instruction=False,
    )
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

    benchmark_start_time = time.monotonic()
    traffic_scheduler.reset_reference_time()

    evaluator = build_evaluator(
        benchmark_config,
        seed_manager=seed_manager,
        session_generator=session_generator,
        benchmark_start_time=benchmark_start_time,
    )
    capturing = CapturingEvaluator(evaluator)

    _run_main_loop(
        session_generator=session_generator,
        traffic_scheduler=traffic_scheduler,
        evaluator=capturing,
        client=client,
        runtime_config=benchmark_config.runtime,
        benchmark_start_time=benchmark_start_time,
    )
    return capturing.results


def _score_run(
    client_config: BaseClientConfig,
    session_config: SyntheticSessionGeneratorConfig,
    server: MockServerHandle,
    *,
    ttfc_ms: float,
    tpoc_ms: float,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> ScoreReport:
    """Drive one client config against a running mock and score the drift."""
    benchmark_config = _build_benchmark_config(
        client_config, session_config, traffic_scheduler, runtime, output_dir
    )
    results = _run_capturing(benchmark_config)
    server_records = server.fetch_records()
    return scorer.score(results, server_records, ttfc_ms=ttfc_ms, tpoc_ms=tpoc_ms)


def _log(kind: str, runtime: RuntimeConfig, server: MockServerHandle) -> None:
    logger.info(
        "Preflight %s: %d sessions vs mock %s",
        kind,
        runtime.max_sessions,
        server.api_base,
    )


# ---------------------------------------------------------------------------
# text group: chat + completions
# ---------------------------------------------------------------------------


def run_text_preflight(
    cfg: PreflightTextCheckConfig,
    *,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> ScoreReport:
    """Chat (streaming SSE) path."""
    with spawn_mock_chat_server(
        ttfc_ms=cfg.server_ttfc_ms,
        tpoc_ms=cfg.server_tpoc_ms,
        num_chunks=cfg.num_response_chunks,
    ) as server:
        _log("text", runtime, server)
        client_config = OpenAIChatCompletionsClientConfig(
            api_base=server.api_base,
            api_key="preflight",
            model="preflight-mock",
            record_preflight_timing=True,
        )
        return _score_run(
            client_config,
            _text_session_config(cfg.input_tokens),
            server,
            ttfc_ms=cfg.server_ttfc_ms,
            tpoc_ms=cfg.server_tpoc_ms,
            traffic_scheduler=traffic_scheduler,
            runtime=runtime,
            output_dir=output_dir,
        )


def run_completions_preflight(
    cfg: PreflightTextCheckConfig,
    *,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> ScoreReport:
    """Completions (non-streaming) path -- one response, so no tpoc metric."""
    with spawn_mock_completions_server(ttfc_ms=cfg.server_ttfc_ms) as server:
        _log("completions", runtime, server)
        client_config = OpenAICompletionsClientConfig(
            api_base=server.api_base,
            api_key="preflight",
            model="preflight-mock",
            record_preflight_timing=True,
        )
        return _score_run(
            client_config,
            _text_session_config(cfg.input_tokens),
            server,
            ttfc_ms=cfg.server_ttfc_ms,
            tpoc_ms=0.0,
            traffic_scheduler=traffic_scheduler,
            runtime=runtime,
            output_dir=output_dir,
        )


# ---------------------------------------------------------------------------
# tts group: tts (HTTP) + realtime_tts + vajra (WS)
# ---------------------------------------------------------------------------


def run_tts_preflight(
    cfg: PreflightTtsCheckConfig,
    *,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> ScoreReport:
    """TTS (HTTP streaming raw audio) path."""
    with spawn_mock_tts_server(
        ttfc_ms=cfg.server_ttfc_ms,
        tpoc_ms=cfg.server_tpoc_ms,
        num_chunks=cfg.num_response_chunks,
        chunk_bytes=_TTS_CHUNK_BYTES,
    ) as server:
        _log("tts", runtime, server)
        client_config = TTSClientConfig(
            api_base=server.api_base,
            api_key="preflight",
            model="preflight-mock",
            voice_id="preflight",
            raw_pcm=True,
            chunk_size=_TTS_CHUNK_BYTES,
            record_preflight_timing=True,
        )
        return _score_run(
            client_config,
            _text_session_config(cfg.input_tokens),
            server,
            ttfc_ms=cfg.server_ttfc_ms,
            tpoc_ms=cfg.server_tpoc_ms,
            traffic_scheduler=traffic_scheduler,
            runtime=runtime,
            output_dir=output_dir,
        )


def run_realtime_tts_preflight(
    cfg: PreflightTtsCheckConfig,
    *,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> ScoreReport:
    """Realtime-TTS (WebSocket) path."""
    with spawn_mock_realtime_tts_server(
        ttfc_ms=cfg.server_ttfc_ms,
        tpoc_ms=cfg.server_tpoc_ms,
        num_chunks=cfg.num_response_chunks,
    ) as server:
        _log("realtime_tts", runtime, server)
        client_config = RealtimeTTSClientConfig(
            api_base=server.api_base,
            api_key="preflight",
            model="preflight-mock",
            voice_id="preflight",
            pacing=_stream_pacing(cfg.input_pacing_tps, cfg.input_chunk_tokens),
            record_preflight_timing=True,
        )
        return _score_run(
            client_config,
            _text_session_config(cfg.input_tokens),
            server,
            ttfc_ms=cfg.server_ttfc_ms,
            tpoc_ms=cfg.server_tpoc_ms,
            traffic_scheduler=traffic_scheduler,
            runtime=runtime,
            output_dir=output_dir,
        )


def run_vajra_tts_preflight(
    cfg: PreflightTtsCheckConfig,
    *,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> ScoreReport:
    """Vajra TTS-stream (WebSocket binary PCM) path."""
    with spawn_mock_vajra_tts_server(
        ttfc_ms=cfg.server_ttfc_ms,
        tpoc_ms=cfg.server_tpoc_ms,
        num_chunks=cfg.num_response_chunks,
    ) as server:
        _log("vajra_tts_stream", runtime, server)
        client_config = VajraTTSStreamClientConfig(
            api_base=server.api_base,
            api_key="preflight",
            model="preflight-mock",
            voice_id="preflight",
            pacing=_stream_pacing(cfg.input_pacing_tps, cfg.input_chunk_tokens),
            record_preflight_timing=True,
        )
        return _score_run(
            client_config,
            _text_session_config(cfg.input_tokens),
            server,
            ttfc_ms=cfg.server_ttfc_ms,
            tpoc_ms=cfg.server_tpoc_ms,
            traffic_scheduler=traffic_scheduler,
            runtime=runtime,
            output_dir=output_dir,
        )


# ---------------------------------------------------------------------------
# stt group
# ---------------------------------------------------------------------------


def run_stt_preflight(
    cfg: PreflightSttCheckConfig,
    *,
    traffic_scheduler: BaseTrafficConfig,
    runtime: RuntimeConfig,
    output_dir: str,
) -> ScoreReport:
    """STT (WebSocket audio-in) path -- synthetic audio input, transcript out."""
    with spawn_mock_stt_server(
        ttfc_ms=cfg.server_ttfc_ms,
        tpoc_ms=cfg.server_tpoc_ms,
        num_chunks=cfg.num_response_chunks,
    ) as server:
        _log("stt", runtime, server)
        client_config = STTClientConfig(
            api_base=server.api_base,
            api_key="preflight",
            model="preflight-mock",
            provider="vllm_realtime",
            sample_rate=cfg.sample_rate,
            ws_chunk_size=cfg.input_chunk_bytes,
            ws_realtime_pacing=True,
            record_preflight_timing=True,
        )
        return _score_run(
            client_config,
            _audio_session_config(cfg.input_seconds, cfg.sample_rate),
            server,
            ttfc_ms=cfg.server_ttfc_ms,
            tpoc_ms=cfg.server_tpoc_ms,
            traffic_scheduler=traffic_scheduler,
            runtime=runtime,
            output_dir=output_dir,
        )
