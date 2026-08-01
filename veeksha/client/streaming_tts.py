"""Provider-agnostic WebSocket client for streaming text-to-speech."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol
from urllib.parse import quote, urlencode, urljoin
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from veeksha.client.base import BaseLLMClient
from veeksha.client.utils import (
    WS_TRANSPORT_ERROR_PRIORITY,
    TextDeltaPacer,
    flatten_ws_exception,
    map_ws_transport_error,
    resolve_provider_api_key,
    segment_text,
    to_websocket_url,
)
from veeksha.core.audio_contract import AudioMetricKey, pcm_bytes_to_duration_ms
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import StreamingTTSClientConfig

logger = init_logger(__name__)
__all__ = ["StreamingTTSClient"]


@dataclass(frozen=True)
class StreamingProtocolEvent:
    """Transport-independent event produced by a provider strategy.

    ``audio`` carries the provider's own wire representation -- raw PCM bytes
    for binary-frame protocols, the undecoded base64 string for protocols that
    wrap audio in a JSON envelope. Turning a request's payloads into PCM is
    :meth:`join_audio`, which the client runs once per request after the stream
    completes, so no decode happens on the event loop that is simultaneously
    pacing text sends.

    Subclasses supply the representation; this class only states the contract.
    """

    audio: bytes | str = b""
    ready: bool = False
    response_started: bool = False
    audio_done: bool = False
    terminal: bool = False
    sample_rate: int | None = None
    error: str | None = None

    @property
    def audio_nbytes(self) -> int:
        """Decoded byte count of ``audio``, without decoding it."""
        raise NotImplementedError

    @staticmethod
    def join_audio(payloads: list[Any]) -> tuple[bytes, str | None]:
        """Concatenate one request's ``audio`` payloads into PCM.

        Returns ``(content, error)``. Runs off the event loop, so it must not
        touch anything request-scoped beyond ``payloads``.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class RawAudioEvent(StreamingProtocolEvent):
    """Event from a protocol whose audio arrives as binary PCM frames."""

    audio: bytes = b""

    @property
    def audio_nbytes(self) -> int:
        return len(self.audio)

    @staticmethod
    def join_audio(payloads: list[bytes]) -> tuple[bytes, str | None]:
        return b"".join(payloads), None


@dataclass(frozen=True)
class Base64AudioEvent(StreamingProtocolEvent):
    """Event from a protocol whose audio arrives base64-encoded in JSON."""

    audio: str = ""

    @property
    def audio_nbytes(self) -> int:
        return _b64_decoded_size(self.audio)

    @staticmethod
    def join_audio(payloads: list[str]) -> tuple[bytes, str | None]:
        """Decode in order and concatenate.

        On a malformed payload the audio decoded so far is still returned
        alongside the error, matching the inline decode this replaced: a bad
        chunk fails the request but does not discard the chunks before it.
        """
        decoded: list[bytes] = []
        for payload in payloads:
            try:
                decoded.append(base64.b64decode(payload, validate=True))
            except (ValueError, TypeError) as exc:
                return b"".join(decoded), f"Invalid base64 audio: {exc}"
        return b"".join(decoded), None


class StreamingTTSError(Exception):
    """Fatal provider event received after a WebSocket handshake."""


class _ClientAbort(Exception):
    """Signal a deliberate mid-stream client hang-up.

    This is an expected adversarial outcome, not a server or transport error.
    """


class StreamingProviderProtocol(Protocol):
    """Provider-specific wire behavior behind the shared streaming lifecycle."""

    provider: str
    protocol_name: str
    default_api_key_env: str
    requires_api_key: bool
    has_ready_event: bool
    explicit_response_trigger: bool
    raw_pcm: bool
    event_class: ClassVar[type[StreamingProtocolEvent]]

    def __init__(
        self, config: StreamingTTSClientConfig, api_key: str | None
    ) -> None: ...

    def build_ws_url(self, api_base: str) -> str: ...

    def headers(self) -> dict[str, str]: ...

    def initial_messages(self) -> list[str]: ...

    def text_message(self, text: str) -> str: ...

    def response_trigger_message(self) -> str | None: ...

    def finish_messages(self) -> list[str]: ...

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent: ...


class _ImplicitResponseProtocol:
    """Common scheduling behavior for providers that synthesize on text receipt."""

    explicit_response_trigger = False
    raw_pcm = True

    def response_trigger_message(self) -> None:
        return None


def _b64_decoded_size(encoded: str) -> int:
    """Exact decoded length of padded standard base64, without decoding it.

    ``audio_chunk_timestamps`` reports a per-chunk byte count that the evaluator
    consumes, so this has to agree with the eventual decode exactly rather than
    approximately. A malformed payload gives a meaningless answer here, but it
    also fails in :meth:`Base64AudioEvent.join_audio`, which surfaces the error.
    """
    length = len(encoded)
    if length == 0:
        return 0
    if encoded.endswith("=="):
        padding = 2
    elif encoded.endswith("="):
        padding = 1
    else:
        padding = 0
    return (length // 4) * 3 - padding


def _decode_json_object(raw: str | bytes) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        return None
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return event if isinstance(event, dict) else None


def _extract_realtime_sample_rate(session: dict[str, Any]) -> int | None:
    audio = session.get("audio")
    if not isinstance(audio, dict):
        return None
    output = audio.get("output")
    if not isinstance(output, dict):
        return None
    audio_format = output.get("format")
    if not isinstance(audio_format, dict):
        return None
    sample_rate = audio_format.get("rate")
    return sample_rate if isinstance(sample_rate, int) else None


def _realtime_error_message(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message

    response = event.get("response")
    if isinstance(response, dict):
        details = response.get("status_details")
        if isinstance(details, dict):
            detail_error = details.get("error")
            if isinstance(detail_error, dict):
                message = detail_error.get("message")
                if isinstance(message, str) and message:
                    return message
            reason = details.get("reason")
            if isinstance(reason, str) and reason:
                return reason
        status = response.get("status")
        if isinstance(status, str) and status:
            return f"Realtime response ended with status {status}"

    return json.dumps(event)


class OpenAIRealtimeProtocol:
    provider = "openai"
    protocol_name = "realtime"
    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = False
    has_ready_event = True
    explicit_response_trigger = True
    raw_pcm = True
    event_class: ClassVar[type[StreamingProtocolEvent]] = Base64AudioEvent

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        normalized = to_websocket_url(api_base).rstrip("/")
        path = "realtime" if normalized.endswith("/v1") else "v1/realtime"
        url = urljoin(f"{normalized}/", path)
        return f"{url}?model={quote(self.config.model, safe='')}"

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def initial_messages(self) -> list[str]:
        output: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": self.config.sample_rate}
        }
        if self.config.voice_id:
            output["voice"] = self.config.voice_id
        session = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {"output": output},
        }
        return [json.dumps({"type": "session.update", "session": session})]

    def text_message(self, text: str) -> str:
        return json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def response_trigger_message(self) -> str:
        return json.dumps(
            {
                "type": "response.create",
                "response": {"output_modalities": ["audio"]},
            }
        )

    def finish_messages(self) -> list[str]:
        return []

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        event = _decode_json_object(raw)
        if event is None:
            return self.event_class()
        event_type = event.get("type")
        if event_type == "session.updated":
            session = event.get("session")
            sample_rate = (
                _extract_realtime_sample_rate(session)
                if isinstance(session, dict)
                else None
            )
            return self.event_class(ready=True, sample_rate=sample_rate)
        if event_type == "response.created":
            return self.event_class(response_started=True)
        if event_type == "response.output_audio.delta":
            encoded_audio = event.get("delta")
            if not isinstance(encoded_audio, str) or not encoded_audio:
                return self.event_class()
            return self.event_class(audio=encoded_audio)
        if event_type == "response.output_audio.done":
            return self.event_class(audio_done=True)
        if event_type == "response.done":
            response = event.get("response")
            if not isinstance(response, dict) or response.get("status") != "completed":
                return self.event_class(error=_realtime_error_message(event))
            return self.event_class(terminal=True)
        if event_type == "error":
            return self.event_class(error=_realtime_error_message(event))
        return self.event_class()


class VajraStreamingProtocol(_ImplicitResponseProtocol):
    provider = "vajra"
    protocol_name = "native_streaming_text"
    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = False
    has_ready_event = True
    event_class: ClassVar[type[StreamingProtocolEvent]] = RawAudioEvent

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        normalized = to_websocket_url(api_base).rstrip("/")
        path = (
            "audio/speech/stream"
            if normalized.endswith("/v1")
            else "v1/audio/speech/stream"
        )
        return urljoin(f"{normalized}/", path)

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def initial_messages(self) -> list[str]:
        message: dict[str, Any] = {
            "type": "session.config",
            "response_format": "pcm",
            "stream_audio": True,
        }
        if self.config.voice_id:
            message["voice"] = self.config.voice_id
        if self.config.language is not None:
            message["language"] = self.config.language
        if self.config.instructions is not None:
            message["instructions"] = self.config.instructions
        if self.config.task_type is not None:
            message["task_type"] = self.config.task_type
        return [json.dumps(message)]

    def text_message(self, text: str) -> str:
        return json.dumps({"type": "input.text", "text": text})

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "input.done"})]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return self.event_class(audio=raw)
        event = _decode_json_object(raw)
        if event is None:
            return self.event_class()
        event_type = event.get("type")
        if event_type == "audio.start":
            sample_rate = event.get("sample_rate")
            return self.event_class(
                ready=True,
                response_started=True,
                sample_rate=sample_rate if isinstance(sample_rate, int) else None,
            )
        if event_type == "audio.done":
            if event.get("error"):
                return self.event_class(
                    error="Vajra TTS stream reported audio.done with error=true"
                )
            return self.event_class(audio_done=True)
        if event_type == "session.done":
            return self.event_class(terminal=True)
        if event_type == "error":
            message = event.get("message")
            return self.event_class(
                error=message if isinstance(message, str) else json.dumps(event)
            )
        return self.event_class()


class ElevenLabsStreamingProtocol(_ImplicitResponseProtocol):
    provider = "elevenlabs"
    protocol_name = "v1_stream_input"
    default_api_key_env = "ELEVENLABS_API_KEY"
    requires_api_key = True
    has_ready_event = False
    event_class: ClassVar[type[StreamingProtocolEvent]] = Base64AudioEvent

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        path = f"v1/text-to-speech/{quote(self.config.voice_id, safe='')}/stream-input"
        query = urlencode(
            {
                "model_id": self.config.model,
                "output_format": f"pcm_{self.config.sample_rate}",
                "auto_mode": str(self.config.auto_mode).lower(),
                "apply_text_normalization": self.config.apply_text_normalization,
            }
        )
        return f"{to_websocket_url(urljoin(normalized, path))}?{query}"

    def headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    def initial_messages(self) -> list[str]:
        payload: dict[str, Any] = {
            "text": " ",
            "voice_settings": {
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost,
                "speed": self.config.speed,
            },
        }
        if not self.config.auto_mode:
            payload["generation_config"] = {
                "chunk_length_schedule": self.config.chunk_length_schedule
            }
        return [json.dumps(payload)]

    def text_message(self, text: str) -> str:
        return json.dumps({"text": text})

    def finish_messages(self) -> list[str]:
        return [json.dumps({"text": ""})]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return self.event_class(error="Unexpected binary ElevenLabs frame")
        event = _decode_json_object(raw)
        if event is None:
            return self.event_class()
        error = event.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or json.dumps(error)
            else:
                message = str(error)
            return self.event_class(error=message)
        encoded_audio = event.get("audio")
        if not isinstance(encoded_audio, str):
            encoded_audio = ""
        terminal = bool(event.get("isFinal"))
        return self.event_class(
            audio=encoded_audio,
            response_started=bool(encoded_audio),
            audio_done=terminal,
            terminal=terminal,
        )


class CartesiaStreamingProtocol(_ImplicitResponseProtocol):
    """Cartesia context protocol with incremental transcript appends."""

    provider = "cartesia"
    protocol_name = "tts_websocket_context"
    default_api_key_env = "CARTESIA_API_KEY"
    requires_api_key = True
    has_ready_event = False
    event_class: ClassVar[type[StreamingProtocolEvent]] = Base64AudioEvent

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""
        self.context_id = str(uuid4())

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        return to_websocket_url(urljoin(normalized, "tts/websocket"))

    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": self.config.cartesia_version,
        }

    def initial_messages(self) -> list[str]:
        return []

    def _generation_message(self, transcript: str, *, continuing: bool) -> str:
        payload: dict[str, Any] = {
            "model_id": self.config.model,
            "transcript": transcript,
            "voice": {"mode": "id", "id": self.config.voice_id},
            "language": self.config.language or "en",
            "context_id": self.context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.config.sample_rate,
            },
            "continue": continuing,
        }
        if self.config.max_buffer_delay_ms is not None:
            payload["max_buffer_delay_ms"] = self.config.max_buffer_delay_ms
        return json.dumps(payload)

    def text_message(self, text: str) -> str:
        return self._generation_message(text, continuing=True)

    def finish_messages(self) -> list[str]:
        return [self._generation_message("", continuing=False)]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return self.event_class(error="Unexpected binary Cartesia frame")
        event = _decode_json_object(raw)
        if event is None:
            return self.event_class()
        event_type = event.get("type")
        if event_type == "error":
            message = (
                event.get("message")
                or event.get("title")
                or event.get("error_code")
                or json.dumps(event)
            )
            return self.event_class(error=str(message))
        if event_type == "chunk":
            encoded_audio = event.get("data")
            if not isinstance(encoded_audio, str) or not encoded_audio:
                return self.event_class(error="Cartesia chunk omitted audio data")
            return self.event_class(audio=encoded_audio, response_started=True)
        if event_type == "done" or event.get("done") is True:
            return self.event_class(audio_done=True, terminal=True)
        return self.event_class()


class _DeepgramStreamingProtocol(_ImplicitResponseProtocol):
    provider = "deepgram"
    default_api_key_env = "DEEPGRAM_API_KEY"
    requires_api_key = True
    event_class: ClassVar[type[StreamingProtocolEvent]] = RawAudioEvent
    endpoint = ""
    ready_event: str | None = None
    response_started_event: str | None = None
    terminal_event = ""

    def __init__(self, config: StreamingTTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def _query_parameters(self) -> dict[str, str | int | float]:
        return {
            "model": self.config.model,
            "encoding": "linear16",
            "sample_rate": self.config.sample_rate,
            "mip_opt_out": str(self.config.mip_opt_out).lower(),
        }

    def build_ws_url(self, api_base: str) -> str:
        normalized = api_base.rstrip("/") + "/"
        query = urlencode(self._query_parameters())
        return f"{to_websocket_url(urljoin(normalized, self.endpoint))}?{query}"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def initial_messages(self) -> list[str]:
        return []

    def text_message(self, text: str) -> str:
        return json.dumps({"type": "Speak", "text": text})

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "Flush"})]

    def parse(self, raw: str | bytes) -> StreamingProtocolEvent:
        if isinstance(raw, bytes):
            return self.event_class(audio=raw)
        event = _decode_json_object(raw)
        if event is None:
            return self.event_class()
        event_type = event.get("type")
        if event_type == "Error":
            return self.event_class(
                error=str(event.get("description") or event.get("code") or event)
            )
        terminal = event_type == self.terminal_event
        return self.event_class(
            ready=event_type == self.ready_event,
            response_started=event_type == self.response_started_event,
            audio_done=terminal,
            terminal=terminal,
        )


class DeepgramFluxStreamingProtocol(_DeepgramStreamingProtocol):
    protocol_name = "v2_flux_speak"
    endpoint = "v2/speak"
    has_ready_event = True
    ready_event = "Connected"
    response_started_event = "SpeechStarted"
    terminal_event = "SpeechMetadata"


class DeepgramAuraStreamingProtocol(_DeepgramStreamingProtocol):
    protocol_name = "v1_aura_speak"
    endpoint = "v1/speak"
    has_ready_event = False
    terminal_event = "Flushed"

    def _query_parameters(self) -> dict[str, str | int | float]:
        parameters = super()._query_parameters()
        parameters["speed"] = self.config.speed
        return parameters


_STREAMING_PROTOCOLS: dict[str, type[StreamingProviderProtocol]] = {
    "openai_realtime": OpenAIRealtimeProtocol,
    "vajra": VajraStreamingProtocol,
    "elevenlabs": ElevenLabsStreamingProtocol,
    "cartesia": CartesiaStreamingProtocol,
    "deepgram_flux": DeepgramFluxStreamingProtocol,
    "deepgram_aura": DeepgramAuraStreamingProtocol,
}


_ERROR_PRIORITY: tuple[type[BaseException], ...] = (
    StreamingTTSError,
    *WS_TRANSPORT_ERROR_PRIORITY,
)


_ABORT_ERROR_PRIORITY: tuple[type[BaseException], ...] = (
    _ClientAbort,
    *_ERROR_PRIORITY,
)


def _map_error(exc: BaseException) -> tuple[int, str]:
    if isinstance(exc, StreamingTTSError):
        return 500, str(exc)
    if isinstance(exc, InvalidStatus):
        body = exc.response.body
        detail = ""
        if isinstance(body, bytes) and body:
            detail = body.decode("utf-8", errors="replace")[:500]
        elif body:
            detail = str(body)[:500]
        message = str(exc) if not detail else f"{exc}: {detail}"
        return exc.response.status_code, message
    return map_ws_transport_error(exc, "Streaming TTS request timed out")


def _round_ms(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


class StreamingTTSClient(BaseLLMClient):
    """Paced text-in/audio-out WebSocket TTS client for every provider."""

    def __init__(self, config: StreamingTTSClientConfig, **kwargs: Any) -> None:
        protocol_class = _STREAMING_PROTOCOLS.get(config.provider)
        if protocol_class is None:
            raise ValueError(f"Unsupported streaming TTS provider: {config.provider}")
        super().__init__(config)
        self.api_key = resolve_provider_api_key(
            config.api_key,
            config.api_key_env,
            protocol_class.default_api_key_env,
            required=protocol_class.requires_api_key,
        )
        self._streaming_config = config
        self._protocol = protocol_class(config, self.api_key)

    def _connect(self, extra_headers: dict[str, str] | None = None) -> Any:
        """Open the provider WebSocket.

        ``extra_headers`` is merged over the protocol's own headers, so provider
        auth survives. Empty or None adds nothing.
        """
        open_timeout = min(self._streaming_config.request_timeout, 30)
        headers = self._protocol.headers()
        if extra_headers:
            headers = {**headers, **extra_headers}
        return connect(
            self._protocol.build_ws_url(str(self.api_base)),
            max_size=None,
            compression=None,
            open_timeout=open_timeout,
            additional_headers=headers,
        )

    async def measure_websocket_rtt_ms(self, samples: int = 5) -> list[float]:
        """Measure ping/pong RTT on independent provider WebSocket connections.

        The probe authenticates against the configured streaming-TTS endpoint
        but doesn't send provider application messages or synthesis text.
        Independent connections match the benchmark's one-connection-per-request
        lifecycle while excluding DNS, TCP, TLS, and WebSocket handshake time
        from the returned measurements.
        """
        if samples < 1:
            raise ValueError("samples must be >= 1")

        pong_timeout_s = min(float(self._streaming_config.request_timeout), 10.0)
        rtt_samples_ms: list[float] = []
        for _ in range(samples):
            async with self._connect() as websocket:
                pong_received = await websocket.ping()
                rtt_s = await asyncio.wait_for(
                    pong_received,
                    timeout=pong_timeout_s,
                )
                rtt_samples_ms.append(rtt_s * 1000)
        return rtt_samples_ms

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Callable[[], None] | None = None,
        on_request_dispatched: Callable[[], None] | None = None,
    ) -> RequestResult:
        text_content = request.channels.get(ChannelModality.TEXT)
        if not isinstance(text_content, TextChannelRequestContent):
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No TEXT channel in request for streaming TTS",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        # Protocol instances are request-scoped so providers such as Cartesia can
        # safely retain per-utterance continuation state under concurrency.
        protocol = type(self._protocol)(self._streaming_config, self.api_key)
        pacing = self._streaming_config.pacing
        segments = segment_text(input_text, pacing.tokens_per_delta)
        pacer = TextDeltaPacer(pacing, seed=pacing.seed + request.id)
        input_tokens = text_content.target_prompt_tokens or sum(
            segment.n_tokens for segment in segments
        )

        abort_config = self._streaming_config.abort
        abort_input_after: int | None = None
        abort_audio_ms: float | None = None
        abort_wall_s: float | None = None
        if abort_config.selects(session_id):
            if abort_config.trigger == "input_fraction":
                abort_input_after = max(
                    1, math.ceil(abort_config.value * len(segments))
                )
            elif abort_config.trigger == "audio_ms":
                abort_audio_ms = abort_config.value
            elif abort_config.trigger == "wall_clock_s":
                abort_wall_s = abort_config.value

        # Each protocol appends in its own wire representation -- raw PCM bytes
        # or undecoded base64 -- and its event class resolves them in one pass
        # once the stream completes.
        audio_payloads: list[bytes | str] = []
        audio_chunk_timestamps: list[list[float | int]] = []
        text_delta_timestamps: list[list[float | int]] = []
        ttfc: float | None = None
        ws_connect_latency: float | None = None
        session_ready_offset: float | None = None
        response_trigger_offset: float | None = None
        response_created_offset: float | None = None
        input_complete_offset: float | None = None
        audio_done_offset: float | None = None
        response_done_offset: float | None = None
        sample_rate = self._streaming_config.sample_rate
        sent_fired = False
        aborted = False
        stream_completed = asyncio.Event()
        start = time.monotonic()

        # preflight timing (recorded only when enabled). Only the request id is
        # sent to the server; the scorer joins the two record books by request_id.
        preflight_enabled = getattr(self.config, "record_preflight_timing", False)
        client_sent_at: float | None = start if preflight_enabled else None
        chunk_recv_times: list[float] = []
        input_send_times: list[float] = []  # per paced text segment
        input_send_deadlines: list[float] = []  # intended send instant per segment
        extra_headers = (
            {"X-Veeksha-Request-Id": str(request.id)} if preflight_enabled else None
        )

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if not sent_fired and on_request_sent is not None:
                on_request_sent()
                sent_fired = True

        async def send_loop(websocket: ClientConnection) -> None:
            nonlocal input_complete_offset, response_trigger_offset
            if pacer.initial_delay_s > 0:
                await asyncio.sleep(pacer.initial_delay_s)
            deadline = time.monotonic()
            response_triggered = False
            sent_words = 0

            for segment_idx, segment in enumerate(segments):
                deadline += pacer.next_gap()
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                offset_ms = (time.monotonic() - start) * 1000
                if not protocol.explicit_response_trigger and not response_triggered:
                    # Native streaming providers treat the first real text
                    # append as the synthesis trigger. Record it before the
                    # await so a very fast response cannot race the timestamp.
                    response_trigger_offset = offset_ms
                    response_triggered = True

                await websocket.send(protocol.text_message(segment.text))
                if preflight_enabled:
                    # Actual send vs intended deadline -> pacing error.
                    input_send_times.append(time.monotonic())
                    input_send_deadlines.append(deadline)
                text_delta_timestamps.append([offset_ms, segment.n_chars])
                sent_words += segment.n_tokens

                if (
                    protocol.explicit_response_trigger
                    and not response_triggered
                    and self._streaming_config.input_output_mode == "duplex"
                    and sent_words >= self._streaming_config.duplex_start_after_tokens
                ):
                    response_trigger_offset = (time.monotonic() - start) * 1000
                    trigger_message = protocol.response_trigger_message()
                    if trigger_message is not None:
                        await websocket.send(trigger_message)
                    response_triggered = True

                if (
                    abort_input_after is not None
                    and segment_idx + 1 >= abort_input_after
                ):
                    raise _ClientAbort()

            input_complete_offset = (time.monotonic() - start) * 1000
            if protocol.explicit_response_trigger and not response_triggered:
                response_trigger_offset = (time.monotonic() - start) * 1000
                trigger_message = protocol.response_trigger_message()
                if trigger_message is not None:
                    await websocket.send(trigger_message)
                response_triggered = True

            for message in protocol.finish_messages():
                await websocket.send(message)

        async def recv_loop(websocket: ClientConnection) -> None:
            nonlocal ttfc, session_ready_offset, response_created_offset
            nonlocal audio_done_offset, response_done_offset, sample_rate
            received_audio_ms = 0.0
            while True:
                raw = await websocket.recv()
                # Stamp receipt before any parse/decode work.
                recv_time = time.monotonic()
                wire_offset_ms = (recv_time - start) * 1000
                event = protocol.parse(raw)
                if event.error:
                    raise StreamingTTSError(event.error)
                if event.ready and session_ready_offset is None:
                    session_ready_offset = wire_offset_ms
                if event.response_started and response_created_offset is None:
                    response_created_offset = wire_offset_ms
                if event.sample_rate is not None and event.sample_rate > 0:
                    if event.sample_rate != sample_rate:
                        logger.warning(
                            "%s sent sample_rate=%d (configured %d); "
                            "using the provider value",
                            protocol.provider,
                            event.sample_rate,
                            sample_rate,
                        )
                    sample_rate = event.sample_rate
                if event.audio:
                    playable_offset_ms = (time.monotonic() - start) * 1000
                    audio_payloads.append(event.audio)
                    audio_chunk_timestamps.append(
                        [playable_offset_ms, event.audio_nbytes]
                    )
                    if preflight_enabled:
                        chunk_recv_times.append(recv_time)
                    if ttfc is None:
                        if response_trigger_offset is None:
                            raise StreamingTTSError(
                                "received audio before the synthesis trigger"
                            )
                        ttfc = wire_offset_ms - response_trigger_offset
                    if response_created_offset is None:
                        response_created_offset = wire_offset_ms
                    fire_sent_once()
                    if abort_audio_ms is not None:
                        received_audio_ms += pcm_bytes_to_duration_ms(
                            event.audio_nbytes, sample_rate
                        )
                        if received_audio_ms >= abort_audio_ms:
                            raise _ClientAbort()
                if event.audio_done and audio_done_offset is None:
                    audio_done_offset = wire_offset_ms
                if event.terminal:
                    response_done_offset = wire_offset_ms
                    stream_completed.set()
                    return

        async def abort_watchdog(delay_s: float) -> None:
            try:
                await asyncio.wait_for(stream_completed.wait(), timeout=delay_s)
            except TimeoutError as error:
                raise _ClientAbort() from error

        error_code: int | None = None
        error_msg: str | None = None
        try:
            async with asyncio.timeout(self._streaming_config.request_timeout):
                async with self._connect(extra_headers=extra_headers) as websocket:
                    ws_connect_latency = (time.monotonic() - start) * 1000
                    for message in protocol.initial_messages():
                        await websocket.send(message)
                    if not protocol.has_ready_event:
                        session_ready_offset = (time.monotonic() - start) * 1000
                    if on_request_dispatched is not None:
                        on_request_dispatched()
                    async with asyncio.TaskGroup() as task_group:
                        task_group.create_task(send_loop(websocket))
                        task_group.create_task(recv_loop(websocket))
                        if abort_wall_s is not None:
                            task_group.create_task(abort_watchdog(abort_wall_s))
        except Exception as exc:
            flattened = flatten_ws_exception(exc, _ABORT_ERROR_PRIORITY)
            if isinstance(flattened, _ClientAbort):
                aborted = True
                logger.debug(
                    "%s streaming TTS request_id=%d session_id=%d aborted "
                    "mid-stream (trigger=%s value=%s)",
                    protocol.provider,
                    request.id,
                    session_id,
                    abort_config.trigger,
                    abort_config.value,
                )
            else:
                error_code, error_msg = _map_error(flattened)
                logger.warning(
                    "%s streaming TTS error: (%s) %s",
                    protocol.provider,
                    error_code,
                    error_msg,
                )

        # Stamped before the bulk decode below, so end-to-end latency measures
        # the provider's stream rather than this client's post-processing.
        completed_at = time.monotonic()
        latency_ms = (completed_at - start) * 1000

        audio_content = b""
        if audio_payloads:
            loop = asyncio.get_running_loop()
            audio_content, decode_error = await loop.run_in_executor(
                None, protocol.event_class.join_audio, audio_payloads
            )
            if decode_error is not None and error_code is None:
                error_code = 500
                error_msg = f"{protocol.provider}: {decode_error}"

        if error_code is None and not aborted and not audio_payloads:
            error_code = 502
            error_msg = f"{protocol.provider} completed the TTS stream without audio"
        success = error_code is None and error_msg is None
        fire_sent_once()

        metrics = {
            "audio_task": AudioTask.TTS,
            AudioMetricKey.PROVIDER.value: protocol.provider,
            AudioMetricKey.PROVIDER_MODEL.value: self._streaming_config.model,
            AudioMetricKey.PROVIDER_PROTOCOL.value: protocol.protocol_name,
            AudioMetricKey.TTFC.value: (round(ttfc, 3) if ttfc is not None else None),
            AudioMetricKey.END_TO_END_LATENCY.value: round(latency_ms, 3),
            AudioMetricKey.CHUNK_COUNT.value: len(audio_payloads),
            AudioMetricKey.RAW_PCM.value: protocol.raw_pcm,
            AudioMetricKey.SAMPLE_RATE.value: sample_rate,
            AudioMetricKey.INPUT_CHARS.value: len(input_text),
            AudioMetricKey.INPUT_TOKENS.value: input_tokens,
            AudioMetricKey.INPUT_TEXT.value: input_text,
            AudioMetricKey.TEXT_PACING_UNIT.value: "whitespace_word",
            AudioMetricKey.TEXT_PACING_RATE.value: pacing.tokens_per_second,
            AudioMetricKey.ABORTED.value: aborted,
            AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: text_delta_timestamps,
            AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: audio_chunk_timestamps,
            AudioMetricKey.WS_CONNECT_LATENCY_MS.value: _round_ms(ws_connect_latency),
            AudioMetricKey.SESSION_READY_OFFSET_MS.value: _round_ms(
                session_ready_offset
            ),
            AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value: _round_ms(
                response_trigger_offset
            ),
            AudioMetricKey.RESPONSE_CREATED_OFFSET_MS.value: _round_ms(
                response_created_offset
            ),
            AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: _round_ms(
                input_complete_offset
            ),
            AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: _round_ms(audio_done_offset),
            AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: _round_ms(
                response_done_offset
            ),
        }

        channels: dict[ChannelModality, ChannelResponse] = {}
        if success or audio_payloads or text_delta_timestamps or aborted:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_content,
                metrics=metrics,
            )

        return RequestResult(
            request_id=request.id,
            session_id=session_id,
            session_total_requests=session_total_requests,
            channels=channels,
            success=success,
            error_code=error_code,
            error_msg=error_msg,
            client_completed_at=completed_at,
            client_sent_at=client_sent_at,
            chunk_recv_times=chunk_recv_times if preflight_enabled else None,
            input_send_times=input_send_times if preflight_enabled else None,
            input_send_deadlines=input_send_deadlines if preflight_enabled else None,
        )
