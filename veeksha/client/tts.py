"""Provider-agnostic HTTP text-to-speech client."""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote, urlencode, urljoin

import httpx
import numpy as np

from veeksha.client.base import BaseLLMClient
from veeksha.client.utils import resolve_provider_api_key
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import TTSClientConfig

logger = init_logger(__name__)

__all__ = ["TTSClient"]


@dataclass(frozen=True)
class TTSHTTPRequest:
    """Normalized request emitted by an HTTP provider strategy."""

    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


class TTSProtocolError(Exception):
    """Raised when an HTTP TTS response violates the audio contract."""


class HTTPProviderProtocol(Protocol):
    """Provider-specific request construction behind the shared HTTP lifecycle."""

    provider: str
    protocol_name: str
    default_api_key_env: str
    requires_api_key: bool
    raw_pcm: bool

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None: ...

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest: ...

    def validate_response(self, response: httpx.Response) -> None: ...

    def iter_audio_chunks(
        self, response: httpx.Response, chunk_size: int
    ) -> AsyncIterator[bytes]: ...


class _RawAudioHTTPProtocol:
    """Shared raw-byte response handling for audio HTTP endpoints."""

    _ALLOWED_MEDIA_TYPES = {
        "application/octet-stream",
        "application/x-wav",
        "binary/octet-stream",
    }

    def validate_response(self, response: httpx.Response) -> None:
        content_type = response.headers.get("Content-Type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if (
            media_type
            and not media_type.startswith("audio/")
            and media_type not in self._ALLOWED_MEDIA_TYPES
        ):
            raise TTSProtocolError(
                f"Expected an audio byte response, got Content-Type {content_type!r}"
            )

    async def iter_audio_chunks(
        self, response: httpx.Response, chunk_size: int
    ) -> AsyncIterator[bytes]:
        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
            if chunk:
                yield chunk


def _audio_speech_url(api_base: str) -> str:
    normalized_base = api_base.rstrip("/")
    if not normalized_base.endswith("/v1"):
        normalized_base = f"{normalized_base}/v1"
    return urljoin(f"{normalized_base}/", "audio/speech")


class OpenAIHTTPProtocol(_RawAudioHTTPProtocol):
    provider = "openai"
    protocol_name = "v1_audio_speech"
    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = False

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key
        self.raw_pcm = config.raw_pcm

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return TTSHTTPRequest(
            url=_audio_speech_url(api_base),
            headers=headers,
            payload={
                "model": self.config.model,
                "input": text,
                "voice": self.config.voice_id,
                "response_format": "pcm" if self.config.raw_pcm else "wav",
                "stream": True,
                "stream_format": "audio",
            },
        )


class ElevenLabsHTTPProtocol(_RawAudioHTTPProtocol):
    provider = "elevenlabs"
    protocol_name = "v1_text_to_speech"
    default_api_key_env = "ELEVENLABS_API_KEY"
    requires_api_key = True
    raw_pcm = True

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest:
        normalized = api_base.rstrip("/") + "/"
        path = f"v1/text-to-speech/{quote(self.config.voice_id, safe='')}"
        query = urlencode({"output_format": f"pcm_{self.config.sample_rate}"})
        return TTSHTTPRequest(
            url=f"{urljoin(normalized, path)}?{query}",
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            payload={
                "text": text,
                "model_id": self.config.model,
                "voice_settings": {
                    "stability": self.config.stability,
                    "similarity_boost": self.config.similarity_boost,
                    "speed": self.config.speed,
                },
                "apply_text_normalization": self.config.apply_text_normalization,
            },
        )


class DeepgramFluxHTTPProtocol(_RawAudioHTTPProtocol):
    provider = "deepgram"
    protocol_name = "v2_flux_speak_http"
    default_api_key_env = "DEEPGRAM_API_KEY"
    requires_api_key = True
    raw_pcm = True

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest:
        normalized = api_base.rstrip("/") + "/"
        query = urlencode(
            {
                "model": self.config.model,
                "encoding": "linear16",
                "container": "none",
                "sample_rate": self.config.sample_rate,
                "mip_opt_out": str(self.config.mip_opt_out).lower(),
            }
        )
        return TTSHTTPRequest(
            url=f"{urljoin(normalized, 'v2/speak')}?{query}",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={"text": text},
        )


def _float32_pcm_to_pcm16(raw_audio: bytes) -> bytes:
    """Normalize Mistral's float32 little-endian PCM into Veeksha PCM16."""
    if len(raw_audio) % 4:
        raise TTSProtocolError(
            "Mistral returned a float32 PCM chunk whose size is not divisible by 4"
        )
    samples = np.frombuffer(raw_audio, dtype="<f4")
    if not np.all(np.isfinite(samples)):
        raise TTSProtocolError("Mistral returned non-finite float32 PCM samples")
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class MistralHTTPProtocol:
    """Mistral TTS: complete text in, SSE-streamed float32 PCM out."""

    provider = "mistral"
    protocol_name = "v1_audio_speech_sse"
    default_api_key_env = "MISTRAL_API_KEY"
    requires_api_key = True
    raw_pcm = True

    def __init__(self, config: TTSClientConfig, api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_request(self, api_base: str, text: str) -> TTSHTTPRequest:
        return TTSHTTPRequest(
            url=_audio_speech_url(api_base),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            payload={
                "model": self.config.model,
                "input": text,
                "voice_id": self.config.voice_id,
                "response_format": "pcm",
                "stream": True,
            },
        )

    def validate_response(self, response: httpx.Response) -> None:
        content_type = response.headers.get("Content-Type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != "text/event-stream":
            raise TTSProtocolError(
                "Expected a Mistral text/event-stream response, "
                f"got Content-Type {content_type!r}"
            )

    async def iter_audio_chunks(
        self, response: httpx.Response, chunk_size: int
    ) -> AsyncIterator[bytes]:
        del chunk_size  # SSE events define provider chunk boundaries.
        event_name: str | None = None
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
                continue
            if not line.startswith("data:"):
                continue
            raw_data = line.removeprefix("data:").strip()
            if not raw_data or raw_data == "[DONE]":
                continue
            try:
                event = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                raise TTSProtocolError(f"Invalid Mistral SSE JSON: {exc}") from exc
            if not isinstance(event, dict):
                continue
            event_type = event_name or event.get("type") or event.get("event")
            event_name = None
            event_data = event.get("data")
            payload = event_data if isinstance(event_data, dict) else event
            if event_type == "speech.audio.done":
                break
            if event_type == "error":
                raise TTSProtocolError(
                    str(payload.get("message") or payload.get("error") or event)
                )
            encoded_audio = payload.get("audio_data")
            if event_type != "speech.audio.delta" and not encoded_audio:
                continue
            if not isinstance(encoded_audio, str) or not encoded_audio:
                raise TTSProtocolError("Mistral audio delta omitted audio_data")
            try:
                float32_audio = base64.b64decode(encoded_audio, validate=True)
            except (TypeError, ValueError) as exc:
                raise TTSProtocolError(f"Invalid Mistral audio_data: {exc}") from exc
            pcm16_audio = _float32_pcm_to_pcm16(float32_audio)
            if pcm16_audio:
                yield pcm16_audio


_HTTP_PROTOCOLS: dict[str, type[HTTPProviderProtocol]] = {
    "openai": OpenAIHTTPProtocol,
    "elevenlabs": ElevenLabsHTTPProtocol,
    "deepgram_flux": DeepgramFluxHTTPProtocol,
    "mistral": MistralHTTPProtocol,
}


class TTSClient(BaseLLMClient):
    """Complete-text HTTP TTS client with provider-specific request strategies."""

    def __init__(self, config: TTSClientConfig, **kwargs: Any) -> None:
        protocol_class = _HTTP_PROTOCOLS.get(config.provider)
        if protocol_class is None:
            raise ValueError(f"Unsupported HTTP TTS provider: {config.provider}")
        super().__init__(config)
        self.api_key = resolve_provider_api_key(
            config.api_key,
            config.api_key_env,
            protocol_class.default_api_key_env,
            required=protocol_class.requires_api_key,
        )
        self._http_config = config
        self._protocol = protocol_class(config, self.api_key)
        self._client_storage = threading.local()

    def _get_client(self) -> httpx.AsyncClient:
        if not hasattr(self._client_storage, "client"):
            self._client_storage.client = httpx.AsyncClient(
                timeout=self._http_config.request_timeout
            )
        return self._client_storage.client

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
                error_msg="No TEXT channel in request for TTS",
                client_completed_at=time.monotonic(),
            )

        input_text = text_content.input_text
        provider_request = self._protocol.build_request(str(self.api_base), input_text)
        audio_chunks: list[bytes] = []
        audio_chunk_timestamps: list[list[float | int]] = []
        error_code: int | None = None
        error_msg: str | None = None
        ttfc: float | None = None
        sent_fired = False
        start = time.monotonic()

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if not sent_fired and on_request_sent is not None:
                on_request_sent()
                sent_fired = True

        # preflight timing (recorded only when enabled). Only the request id is
        # sent to the server; the scorer joins the two record books by request_id.
        preflight_enabled = getattr(self.config, "record_preflight_timing", False)
        client_sent_at: float | None = None
        chunk_recv_times: list[float] = []
        if preflight_enabled:
            provider_request.headers["X-Veeksha-Request-Id"] = str(request.id)
            client_sent_at = start

        try:
            async with self._get_client().stream(
                "POST",
                provider_request.url,
                headers=provider_request.headers,
                json=provider_request.payload,
                timeout=self._http_config.request_timeout,
            ) as response:
                response.raise_for_status()
                self._protocol.validate_response(response)
                if on_request_dispatched is not None:
                    on_request_dispatched()
                async for chunk in self._protocol.iter_audio_chunks(
                    response, self._http_config.chunk_size
                ):
                    receive_time = time.monotonic()
                    # Client receipt of each audio chunk.
                    if preflight_enabled:
                        chunk_recv_times.append(receive_time)
                    offset_ms = (receive_time - start) * 1000
                    if ttfc is None:
                        ttfc = offset_ms
                    audio_chunks.append(chunk)
                    audio_chunk_timestamps.append([round(offset_ms, 3), len(chunk)])
                    fire_sent_once()
                if not audio_chunks:
                    raise TTSProtocolError(
                        f"{self._protocol.provider} returned an empty audio response"
                    )
        except TTSProtocolError as exc:
            error_code = 502
            error_msg = str(exc)
        except httpx.HTTPStatusError as exc:
            error_code = exc.response.status_code
            error_msg = str(exc)
        except httpx.ConnectError as exc:
            error_code = 503
            error_msg = str(exc)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = "TTS request timed out"
        except Exception as exc:
            error_code = 520
            error_msg = str(exc)
            logger.exception(
                "%s HTTP TTS error: (%s) %s",
                self._protocol.provider,
                error_code,
                error_msg,
            )

        completed_at = time.monotonic()
        latency_ms = (completed_at - start) * 1000
        success = error_code is None and error_msg is None
        fire_sent_once()

        if not success:
            logger.warning(
                "%s HTTP TTS error: (%s) %s",
                self._protocol.provider,
                error_code,
                error_msg,
            )

        channels: dict[ChannelModality, ChannelResponse] = {}
        if success:
            rounded_latency = round(latency_ms, 3)
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=b"".join(audio_chunks),
                metrics={
                    "audio_task": AudioTask.TTS,
                    AudioMetricKey.PROVIDER.value: self._protocol.provider,
                    AudioMetricKey.PROVIDER_MODEL.value: self._http_config.model,
                    AudioMetricKey.PROVIDER_PROTOCOL.value: (
                        self._protocol.protocol_name
                    ),
                    AudioMetricKey.TTFC.value: (
                        round(ttfc, 3) if ttfc is not None else None
                    ),
                    AudioMetricKey.END_TO_END_LATENCY.value: rounded_latency,
                    AudioMetricKey.CHUNK_COUNT.value: len(audio_chunks),
                    AudioMetricKey.RAW_PCM.value: self._protocol.raw_pcm,
                    AudioMetricKey.SAMPLE_RATE.value: self._http_config.sample_rate,
                    AudioMetricKey.INPUT_CHARS.value: len(input_text),
                    AudioMetricKey.INPUT_TOKENS.value: (
                        text_content.target_prompt_tokens or 0
                    ),
                    AudioMetricKey.INPUT_TEXT.value: input_text,
                    AudioMetricKey.TEXT_DELTA_TIMESTAMPS.value: [
                        [0.0, len(input_text)]
                    ],
                    AudioMetricKey.AUDIO_CHUNK_TIMESTAMPS.value: (
                        audio_chunk_timestamps
                    ),
                    AudioMetricKey.RESPONSE_TRIGGER_OFFSET_MS.value: 0.0,
                    AudioMetricKey.INPUT_COMMIT_OFFSET_MS.value: 0.0,
                    AudioMetricKey.AUDIO_DONE_OFFSET_MS.value: rounded_latency,
                    AudioMetricKey.RESPONSE_DONE_OFFSET_MS.value: rounded_latency,
                },
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
        )
