"""Provider-agnostic WebSocket client for streaming speech-to-text.

All providers stream PCM16 over a WebSocket and report transcription metrics
(TTFC, end-to-end latency, and RTF) through one concrete ``STTClient``
lifecycle. Audio is paced at 1x playback when ``ws_realtime_pacing`` is on, and
the send and receive loops run concurrently. ``provider`` selects an internal
adapter for authentication, URL construction, PCM framing, EOF, and events.

STT accepts audio-file requests but always decodes and streams their PCM over a
WebSocket. There is intentionally no separate HTTP/batch STT client.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence
from urllib.parse import urlencode, urljoin

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from veeksha.client.base import BaseLLMClient
from veeksha.client.utils import (
    map_ws_transport_error,
    resolve_provider_api_key,
    to_websocket_url,
)
from veeksha.core.audio_contract import AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import AudioChannelRequestContent
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import STTClientConfig

logger = init_logger(__name__)

__all__ = ["STTClient"]

BYTES_PER_SAMPLE = 2
TranscriptSnapshotRow = dict[str, float | str]

# Voxtral streaming tokens injected by the model that should be stripped.
_STREAMING_TOKEN_RE = re.compile(r"\[STREAMING_(?:PAD|WORD)\]")


def _pcm_duration_ms(pcm_bytes: int, sample_rate: int) -> float:
    """Compute PCM16 mono audio duration in ms."""
    return (pcm_bytes / BYTES_PER_SAMPLE / sample_rate) * 1000


def _clean_transcript(text: str) -> str:
    """Strip Voxtral streaming control tokens and collapse whitespace."""
    text = _STREAMING_TOKEN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _warm_audio_stack(target_sr: int) -> None:
    """Decode a scratch clip so the first request does not pay for it.

    The first ``librosa.load`` call is expensive: it imports
    ``librosa.core.audio`` and its dependencies. That would otherwise land
    under the per-clip lock and stall every request in flight at benchmark
    start.
    """
    try:
        import io

        import librosa
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(
            buf,
            np.zeros(256, dtype="float32"),
            target_sr,
            format="WAV",
            subtype="PCM_16",
        )
        buf.seek(0)
        librosa.load(buf, sr=target_sr, mono=True)
    except Exception:  # noqa: BLE001 - warming is best effort
        pass


def _audio_to_pcm16_bytes(audio_path: str, target_sr: int) -> bytes:
    """Load an audio file and convert to raw PCM16 bytes at target sample rate."""
    import librosa

    audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


def _metadata_ms(metadata: dict, key: str) -> Optional[float]:
    """Read an optional millisecond offset from request metadata."""
    value = metadata.get(key)
    if value is None:
        return None
    return float(value)


def _slice_pcm16_bytes(
    pcm_bytes: bytes | memoryview,
    sample_rate: int,
    *,
    start_ms: Optional[float],
    end_ms: Optional[float],
) -> bytes | memoryview:
    """Slice raw PCM16 mono bytes by millisecond offsets.

    Accepts a ``memoryview`` for zero-copy slicing of cached clip PCM (the
    slice then aliases the cached buffer instead of duplicating ~minutes of
    audio per session).
    """
    if start_ms is None and end_ms is None:
        return pcm_bytes

    total_samples = len(pcm_bytes) // BYTES_PER_SAMPLE
    start_sample = 0 if start_ms is None else int(round(start_ms * sample_rate / 1000))
    end_sample = (
        total_samples if end_ms is None else int(round(end_ms * sample_rate / 1000))
    )

    if start_sample < 0:
        raise ValueError(f"input_audio_start_ms must be non-negative; got {start_ms}")
    if end_sample <= start_sample:
        raise ValueError(
            "input_audio_end_ms must be greater than input_audio_start_ms; "
            f"got start_ms={start_ms}, end_ms={end_ms}"
        )
    if end_sample > total_samples:
        raise ValueError(
            "Requested audio slice exceeds decoded clip length: "
            f"end_ms={end_ms}, clip_ms={_pcm_duration_ms(len(pcm_bytes), sample_rate):.3f}"
        )

    start_byte = start_sample * BYTES_PER_SAMPLE
    end_byte = end_sample * BYTES_PER_SAMPLE
    return pcm_bytes[start_byte:end_byte]


# Decoded clips (and their pre-encoded wire messages) are cached per client;
# traces replay a finite clip set, so this is bounded in practice. The FIFO
# cap guards pathological manifests with thousands of distinct clips.
_CLIP_CACHE_MAX_CLIPS = 64


@dataclass(frozen=True)
class _ClipAssets:
    """Deterministic per-clip artifacts shared by every session replaying it.

    ``wire_messages[i]`` is the provider append-message for a non-final chunk
    ``pcm[i * chunk_size : (i + 1) * chunk_size]``; the trailing partial chunk
    (from the clip end or a per-session slice) is encoded on the fly.
    """

    pcm: bytes
    wire_messages: list[str | bytes]


@dataclass
class STTStreamResult:
    """Provider-neutral transcript and timing output from one streaming request.

    All timings are in milliseconds, measured with ``time.monotonic`` deltas:

    - ``ttfc`` (time to first content): from the first audio byte on the wire
      to the first delta whose own payload still contains text after
      control-token cleaning. Empty progress/keepalive deltas and pure
      control-token pads do not count. Falls back to the completion message
      when the stream produces a final transcript without any content-bearing
      delta.
    - ``time_to_first_visible_text``: from the first audio byte on the wire to
      the first moment the assembled (concatenated then cleaned) transcript is
      non-empty, i.e. when a user watching the live transcript would first see
      text. Matches ``ttfc`` for well-formed streams; the two differ when
      cleaning a delta in isolation disagrees with cleaning the assembled
      transcript (e.g. a control token split across deltas cleans non-empty on
      its own but vanishes once joined). Same completion fallback as ``ttfc``.
    - ``time_to_first_partial``: from end-of-audio (EOF sentinel sent) to the
      first delta after EOF with a non-empty assembled transcript; ``None``
      when no such delta arrives before completion.
    - ``time_to_final_transcript``: from end-of-audio to the completion
      message.
    """

    ttfc: Optional[float]
    time_to_first_visible_text: Optional[float]
    time_to_first_partial: Optional[float]
    time_to_final_transcript: Optional[float]
    partial_transcript: Optional[str]
    final_transcript: str
    transcript_snapshots: list[TranscriptSnapshotRow]
    chunk_count: int
    pcm_byte_count: int
    # preflight timing (client record book, populated only when enabled)
    client_sent_at: Optional[float] = None
    chunk_recv_times: Optional[list[float]] = None
    input_send_times: Optional[list[float]] = None
    input_send_deadlines: Optional[list[float]] = None


class TranscriptSnapshotRecorder:
    """Records evolving transcripts relative to first sent audio byte."""

    def __init__(self) -> None:
        self._audio_started_at: Optional[float] = None
        self._snapshots: list[TranscriptSnapshotRow] = []

    @property
    def snapshots(self) -> list[TranscriptSnapshotRow]:
        return self._snapshots

    def mark_audio_started(self, now: float) -> None:
        if self._audio_started_at is None:
            self._audio_started_at = now

    def add(self, now: float, transcript: str) -> None:
        if not transcript:
            return
        if self._snapshots and self._snapshots[-1]["transcript"] == transcript:
            return
        self._snapshots.append(
            {
                "elapsed_ms": round(self._elapsed_ms(now), 3),
                "transcript": transcript,
            }
        )

    def _elapsed_ms(self, now: float) -> float:
        assert (
            self._audio_started_at is not None
        ), "snapshot recorded before any audio was sent"
        return (now - self._audio_started_at) * 1000


class STTClient(BaseLLMClient):
    """Streaming PCM-in/text-out WebSocket lifecycle for every STT provider.

    Provider differences are isolated in ``_STTProviderProtocol`` strategies.
    """

    def __init__(self, config: "STTClientConfig", **kwargs: Any) -> None:
        protocol_class = _STT_PROTOCOLS.get(config.provider)
        if protocol_class is None:
            raise ValueError(f"Unsupported STT provider: {config.provider}")
        super().__init__(config)
        api_key = resolve_provider_api_key(
            config.api_key,
            config.api_key_env,
            protocol_class.default_api_key_env,
            required=protocol_class.requires_api_key,
        )
        self._protocol = protocol_class(config, api_key)
        self._model = config.model
        self._sample_rate = config.sample_rate
        self._ws_chunk_size = config.ws_chunk_size
        self._pacing = config.ws_realtime_pacing
        self._request_timeout = config.request_timeout
        self._ws_ping_interval_s = config.ws_ping_interval_s
        self._ws_ping_timeout_s = config.ws_ping_timeout_s
        self._ws_compression = "deflate" if config.ws_permessage_deflate else None
        self._ws_url = self._protocol.build_ws_url(str(self.api_base or ""))

        # Per-clip decode + wire-message cache, shared across the worker
        # threads that all hold this one client instance.
        self._clip_cache: OrderedDict[str, _ClipAssets] = OrderedDict()
        self._clip_cache_lock = threading.Lock()
        self._clip_locks: dict[str, threading.Lock] = {}

        _warm_audio_stack(self._sample_rate)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _connect(self, extra_headers: dict[str, str] | None = None) -> Any:
        """Open the provider WebSocket.

        ``extra_headers`` is merged over the protocol's own headers, so provider
        auth survives. Empty or None adds nothing.
        """
        headers = self._protocol.headers()
        if extra_headers:
            headers = {**headers, **extra_headers}
        return connect(
            self._ws_url,
            max_size=None,
            ping_interval=self._ws_ping_interval_s,
            ping_timeout=self._ws_ping_timeout_s,
            compression=self._ws_compression,
            additional_headers=headers,
        )

    async def _maybe_pace_until(self, target_at: float) -> None:
        """Sleep until an absolute audio schedule time when pacing is on."""
        if not self._pacing:
            return
        delay_s = target_at - time.monotonic()
        if delay_s > 0:
            await asyncio.sleep(delay_s)

    def _clip_lock(self, audio_path: str) -> threading.Lock:
        """Return the per-clip lock guarding decode/encode for one path."""
        with self._clip_cache_lock:
            lock = self._clip_locks.get(audio_path)
            if lock is None:
                lock = self._clip_locks[audio_path] = threading.Lock()
            return lock

    def _clip_assets(self, audio_path: str) -> _ClipAssets:
        """Decode a clip and pre-encode its append messages, cached per clip.

        Blocking (audio decode + base64/JSON encode); callers on an event
        loop must run this in an executor. The per-clip lock makes the
        thundering herd at benchmark start decode each clip exactly once.
        """
        with self._clip_lock(audio_path):
            with self._clip_cache_lock:
                assets = self._clip_cache.get(audio_path)
            if assets is not None:
                return assets

            pcm = _audio_to_pcm16_bytes(audio_path, self._sample_rate)
            view = memoryview(pcm)
            full_end = len(pcm) - len(pcm) % self._ws_chunk_size
            wire_messages = [
                self._protocol.encode_chunk(
                    view[offset : offset + self._ws_chunk_size], final=False
                )
                for offset in range(0, full_end, self._ws_chunk_size)
            ]
            assets = _ClipAssets(pcm=pcm, wire_messages=wire_messages)

            with self._clip_cache_lock:
                self._clip_cache[audio_path] = assets
                while len(self._clip_cache) > _CLIP_CACHE_MAX_CLIPS:
                    evicted_path, _ = self._clip_cache.popitem(last=False)
                    # Drop the per-path lock with the cache entry so a long
                    # benchmark against many distinct clips does not retain a
                    # lock object per path forever. Skip locks still held so a
                    # concurrent decoder for that path does not race a newly
                    # created lock for the same path.
                    lock = self._clip_locks.get(evicted_path)
                    if lock is not None and not lock.locked():
                        self._clip_locks.pop(evicted_path, None)
            return assets

    async def _stream(
        self,
        pcm_bytes: bytes | memoryview,
        wire_messages: Optional[list[str | bytes]] = None,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
        request_id: Optional[int] = None,
    ) -> STTStreamResult:
        """Stream pre-decoded PCM16 to the provider and collect the transcript.

        The sender and receiver run concurrently so partial transcripts are
        timed when they arrive, not after the (paced) upload completes. ``ttfc``
        is anchored to ``audio_started_at`` (the first audio byte on the wire),
        so clip decode, connect, and handshake latency are excluded.

        ``wire_messages`` are the cached pre-encoded append messages for the
        clip this PCM is a prefix of (chunk ``i`` at byte ``i * chunk_size``);
        the trailing partial chunk is encoded on the fly.
        """
        ttfc: Optional[float] = None
        audio_started_at: Optional[float] = None
        audio_end_at: Optional[float] = None
        time_to_first_visible_text: Optional[float] = None
        time_to_first_partial: Optional[float] = None
        time_to_final_transcript: Optional[float] = None
        partial_transcript: Optional[str] = None
        final_transcript = ""
        chunk_count = 0
        delta_chunks: list[str] = []
        committed_chunks: list[str] = []
        provisional_transcript = ""
        provisional_is_delta = False
        snapshots = TranscriptSnapshotRecorder()

        def assembled_transcript() -> str:
            parts = list(committed_chunks)
            raw_delta_text = "".join(delta_chunks)
            if provisional_is_delta:
                raw_delta_text += provisional_transcript
            delta_text = _clean_transcript(raw_delta_text)
            if delta_text:
                parts.append(delta_text)
            if provisional_transcript and not provisional_is_delta:
                parts.append(provisional_transcript)
            return _clean_transcript(" ".join(parts))

        # Request handed to the transport (always recorded, so the harness
        # lifecycle dispatch-drift metrics cover STT on normal runs too).
        client_sent_at = time.monotonic()

        # preflight timing (recorded only when enabled). Only the request id is
        # sent to the server; the scorer joins the two record books by request_id.
        preflight_enabled = getattr(self.config, "record_preflight_timing", False)
        chunk_recv_times: list[float] = []
        input_send_times: list[float] = []  # per paced audio chunk
        input_send_deadlines: list[float] = []
        extra_headers = (
            {"X-Veeksha-Request-Id": str(request_id)}
            if preflight_enabled and request_id is not None
            else None
        )

        async with self._connect(extra_headers=extra_headers) as ws:
            await self._protocol.open_session(ws, self._model)
            if on_request_dispatched is not None:
                on_request_dispatched()

            async def _send() -> None:
                nonlocal audio_end_at, audio_started_at
                for byte_offset in range(0, len(pcm_bytes), self._ws_chunk_size):
                    if audio_started_at is None:
                        audio_started_at = time.monotonic()
                        snapshots.mark_audio_started(audio_started_at)
                    else:
                        await self._maybe_pace_until(
                            audio_started_at
                            + byte_offset / BYTES_PER_SAMPLE / self._sample_rate
                        )
                    chunk_end = byte_offset + self._ws_chunk_size
                    final_chunk = chunk_end >= len(pcm_bytes)
                    if (
                        wire_messages is not None
                        and chunk_end <= len(pcm_bytes)
                        and not final_chunk
                    ):
                        message = wire_messages[byte_offset // self._ws_chunk_size]
                    else:
                        message = self._protocol.encode_chunk(
                            pcm_bytes[byte_offset:chunk_end], final=final_chunk
                        )
                    await ws.send(message)
                    if preflight_enabled and audio_started_at is not None:
                        # Actual send vs the audio-clock deadline it paces to.
                        input_send_times.append(time.monotonic())
                        input_send_deadlines.append(
                            audio_started_at
                            + byte_offset / BYTES_PER_SAMPLE / self._sample_rate
                        )
                if audio_started_at is not None:
                    await self._maybe_pace_until(
                        audio_started_at
                        + len(pcm_bytes) / BYTES_PER_SAMPLE / self._sample_rate
                    )
                for message in self._protocol.finish_messages():
                    await ws.send(message)
                audio_end_at = time.monotonic()

            send_task = asyncio.ensure_future(_send())
            try:
                while True:
                    try:
                        raw_message = await ws.recv()
                    except ConnectionClosedOK:
                        if not self._protocol.clean_close_is_terminal:
                            raise
                        # Flux may close normally after its final EndOfTurn
                        # instead of emitting a separate Metadata event. Wait
                        # for our sender so an early provider close still
                        # surfaces as a send failure rather than a success.
                        await send_task
                        kind, text = "done", ""
                    else:
                        kind, text = self._protocol.parse_message(
                            _decode_stt_json_message(
                                raw_message, provider=self._protocol.provider
                            )
                        )
                    now = time.monotonic()  # response-chunk receipt
                    if kind in (
                        "delta",
                        "snapshot",
                        "partial_delta",
                        "final_delta",
                        "commit",
                    ):
                        if preflight_enabled:
                            chunk_recv_times.append(now)
                        # TTFC counts only provider events whose own payload
                        # carries transcript text after cleaning; empty
                        # progress/keepalive events and pure control-token
                        # pads are skipped. See
                        # STTStreamResult for how this differs from
                        # time_to_first_visible_text below.
                        if ttfc is None and _clean_transcript(text):
                            assert (
                                audio_started_at is not None
                            ), "delta arrived before any audio was sent"
                            ttfc = (now - audio_started_at) * 1000
                            if on_request_sent is not None:
                                on_request_sent()
                        if kind == "delta":
                            delta_chunks.append(text)
                        elif kind == "snapshot":
                            provisional_transcript = _clean_transcript(text)
                            provisional_is_delta = False
                        elif kind == "partial_delta":
                            provisional_transcript = text
                            provisional_is_delta = True
                        elif kind == "final_delta":
                            delta_chunks.append(text)
                            provisional_transcript = ""
                            provisional_is_delta = False
                        else:
                            committed = _clean_transcript(text)
                            if committed:
                                committed_chunks.append(committed)
                            delta_chunks.clear()
                            provisional_transcript = ""
                            provisional_is_delta = False
                        if _clean_transcript(text):
                            chunk_count += 1
                        current_transcript = assembled_transcript()
                        snapshots.add(now, current_transcript)
                        if time_to_first_visible_text is None and current_transcript:
                            assert (
                                audio_started_at is not None
                            ), "transcript arrived before any audio was sent"
                            time_to_first_visible_text = (now - audio_started_at) * 1000
                        if (
                            audio_end_at is not None
                            and time_to_first_partial is None
                            and current_transcript
                        ):
                            time_to_first_partial = (now - audio_end_at) * 1000
                            partial_transcript = current_transcript
                    elif kind == "done":
                        final_transcript = _clean_transcript(
                            text or assembled_transcript()
                        )
                        snapshots.add(now, final_transcript)
                        if final_transcript:
                            if ttfc is None:
                                assert (
                                    audio_started_at is not None
                                ), "completion arrived before any audio was sent"
                                ttfc = (now - audio_started_at) * 1000
                                if on_request_sent is not None:
                                    on_request_sent()
                            if time_to_first_visible_text is None:
                                assert (
                                    audio_started_at is not None
                                ), "transcript arrived before any audio was sent"
                                time_to_first_visible_text = (
                                    now - audio_started_at
                                ) * 1000
                            if chunk_count == 0:
                                chunk_count = 1
                        if audio_end_at is not None:
                            time_to_final_transcript = (now - audio_end_at) * 1000
                        break
                    elif kind == "error":
                        raise _STTProtocolError(
                            f"{self._protocol.provider} streaming error: {text or 'unknown'}"
                        )
            finally:
                # Normal path: the sender is already done (servers only emit
                # "done" after EOF). Error path: cancel it. A CancelledError is
                # expected; any other error is a real send failure and must
                # propagate.
                if not send_task.done():
                    send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        if not final_transcript:
            final_transcript = assembled_transcript()

        return STTStreamResult(
            ttfc=ttfc,
            time_to_first_visible_text=time_to_first_visible_text,
            time_to_first_partial=time_to_first_partial,
            time_to_final_transcript=time_to_final_transcript,
            partial_transcript=partial_transcript,
            final_transcript=final_transcript,
            transcript_snapshots=snapshots.snapshots,
            chunk_count=chunk_count,
            pcm_byte_count=len(pcm_bytes),
            client_sent_at=client_sent_at,
            chunk_recv_times=chunk_recv_times if preflight_enabled else None,
            input_send_times=input_send_times if preflight_enabled else None,
            input_send_deadlines=input_send_deadlines if preflight_enabled else None,
        )

    # ------------------------------------------------------------------
    # Request lifecycle (shared)
    # ------------------------------------------------------------------

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        """Stream an audio file to the STT API and collect transcription metrics."""

        sent_fired = False
        dispatched_fired = False

        def fire_sent_once() -> None:
            nonlocal sent_fired
            if sent_fired:
                return
            sent_fired = True
            if on_request_sent is not None:
                on_request_sent()

        def fire_dispatched_once() -> None:
            nonlocal dispatched_fired
            if dispatched_fired:
                return
            dispatched_fired = True
            if on_request_dispatched is not None:
                on_request_dispatched()

        def finish_callbacks() -> None:
            # Error results are returned rather than raised, so both dispatch
            # orderings need a fallback to avoid blocking every later ticket.
            fire_dispatched_once()
            fire_sent_once()

        audio_content = request.channels.get(ChannelModality.AUDIO)
        if not isinstance(audio_content, AudioChannelRequestContent):
            finish_callbacks()
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=400,
                error_msg="No AUDIO channel in request for STT",
                client_completed_at=time.monotonic(),
            )

        audio_path = audio_content.input_audio

        try:
            os.path.getsize(audio_path)
        except OSError as e:
            finish_callbacks()
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=404,
                error_msg=f"Failed to read audio file {audio_path}: {e}",
                client_completed_at=time.monotonic(),
            )

        logger.debug(
            "[STT %s] request_id=%d session_id=%d file=%s",
            self._protocol.provider,
            request.id,
            session_id,
            audio_path,
        )

        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        stream_result: Optional[STTStreamResult] = None

        # Decode and slice before the latency clock so file I/O and DSP don't
        # inflate TTFC. Decode and append-message encoding are cached per
        # clip and run in an executor so they never stall the shared event
        # loop (a synchronous decode here freezes pacing for every session
        # on this worker's loop).
        try:
            loop = asyncio.get_running_loop()
            clip = await loop.run_in_executor(None, self._clip_assets, audio_path)
            start_ms = _metadata_ms(request.metadata, "input_audio_start_ms")
            pcm_bytes = _slice_pcm16_bytes(
                memoryview(clip.pcm),
                self._sample_rate,
                start_ms=start_ms,
                end_ms=_metadata_ms(request.metadata, "input_audio_end_ms"),
            )
            # Cached messages are aligned to the clip start; a non-zero slice
            # start shifts the chunk grid, so encode per chunk in that case.
            wire_messages = (
                None if start_ms is not None and start_ms > 0 else clip.wire_messages
            )
        except Exception as e:
            finish_callbacks()
            return RequestResult(
                request_id=request.id,
                session_id=session_id,
                session_total_requests=session_total_requests,
                success=False,
                error_code=422,
                error_msg=f"Failed to decode audio file {audio_path}: {e}",
                client_completed_at=time.monotonic(),
            )

        t_start = time.monotonic()

        try:
            async with asyncio.timeout(self._request_timeout):
                stream_result = await self._stream(
                    pcm_bytes,
                    wire_messages,
                    on_request_sent=fire_sent_once,
                    on_request_dispatched=fire_dispatched_once,
                    request_id=request.id,
                )
        except Exception as exc:
            error_code, error_msg = _map_stt_error(exc, self._request_timeout)
            logger.warning(
                "[STT %s] request_id=%d error (%s): %s",
                self._protocol.provider,
                request.id,
                error_code,
                error_msg,
            )

        finish_callbacks()
        completed_at = time.monotonic()
        total_latency_ms = (completed_at - t_start) * 1000
        success = error_msg is None and error_code is None

        channels = {}
        if success and stream_result is not None:
            input_audio_duration_ms = _pcm_duration_ms(
                stream_result.pcm_byte_count, self._sample_rate
            )
            # Report the input clip's byte count; the evaluator derives
            # duration/RTF from pcm_byte_count + sample_rate.
            metrics_dict: dict[str, Any] = {
                "audio_task": AudioTask.STT,
                AudioMetricKey.PROVIDER.value: self._protocol.provider,
                AudioMetricKey.PROVIDER_MODEL.value: self._model,
                AudioMetricKey.PROVIDER_PROTOCOL.value: self._protocol.protocol_name,
                # Omit inventing 0 ms when no content-bearing event was timed.
                AudioMetricKey.TTFC.value: (
                    round(stream_result.ttfc, 3)
                    if stream_result.ttfc is not None
                    else None
                ),
                AudioMetricKey.END_TO_END_LATENCY.value: round(total_latency_ms, 3),
                "time_to_first_visible_text": (
                    round(stream_result.time_to_first_visible_text, 3)
                    if stream_result.time_to_first_visible_text is not None
                    else None
                ),
                "time_to_first_partial": (
                    round(stream_result.time_to_first_partial, 3)
                    if stream_result.time_to_first_partial is not None
                    else None
                ),
                "time_to_final_transcript": (
                    round(stream_result.time_to_final_transcript, 3)
                    if stream_result.time_to_final_transcript is not None
                    else None
                ),
                AudioMetricKey.CHUNK_COUNT.value: stream_result.chunk_count,
                AudioMetricKey.PCM_BYTE_COUNT.value: stream_result.pcm_byte_count,
                AudioMetricKey.RAW_PCM.value: True,
                AudioMetricKey.INPUT_TOKENS.value: len(
                    stream_result.final_transcript.split()
                ),
                AudioMetricKey.SAMPLE_RATE.value: self._sample_rate,
                "input_audio_duration_ms": round(input_audio_duration_ms, 3),
                "partial_transcript": stream_result.partial_transcript,
                "final_transcript": stream_result.final_transcript,
                "transcript_snapshots": stream_result.transcript_snapshots,
            }

            # Ground truth and dataset metadata are guaranteed by the trace generator.
            for key, value in request.metadata.items():
                metrics_dict.setdefault(key, value)

            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=stream_result.final_transcript,
                metrics=metrics_dict,
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
            client_sent_at=(
                stream_result.client_sent_at if stream_result is not None else None
            ),
            chunk_recv_times=(
                stream_result.chunk_recv_times if stream_result is not None else None
            ),
            input_send_times=(
                stream_result.input_send_times if stream_result is not None else None
            ),
            input_send_deadlines=(
                stream_result.input_send_deadlines
                if stream_result is not None
                else None
            ),
        )


class _STTProtocolError(Exception):
    """Fatal provider event received after a WebSocket handshake."""


def _decode_stt_json_message(
    raw_message: str | bytes, *, provider: str
) -> dict[str, Any]:
    """Parse a provider text frame as a JSON object.

    STT providers in this client speak JSON control/transcript events, so a
    binary frame or non-object JSON is a protocol violation. A bare
    ``json.loads`` already failed the request (``send_request`` catches
    everything and ``map_ws_transport_error`` has a catch-all), but it
    surfaced as an opaque 520 carrying a stringified ``JSONDecodeError``.
    Raising ``_STTProtocolError`` instead classifies these as 500 and names
    the offending provider. It also rejects binary frames, which
    ``json.loads`` would otherwise accept whenever they happen to be UTF-8
    JSON, and rejects non-object JSON before ``parse_message`` calls ``.get``
    on it.
    """
    if isinstance(raw_message, bytes):
        raise _STTProtocolError(
            f"{provider} sent unexpected binary WebSocket frame "
            f"({len(raw_message)} bytes)"
        )
    try:
        message = json.loads(raw_message)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise _STTProtocolError(
            f"{provider} sent non-JSON WebSocket message: {error}"
        ) from error
    if not isinstance(message, dict):
        raise _STTProtocolError(
            f"{provider} sent non-object JSON WebSocket message: "
            f"{type(message).__name__}"
        )
    return message


def _map_stt_error(exc: BaseException, timeout_s: float) -> tuple[int, str]:
    if isinstance(exc, _STTProtocolError):
        return 500, str(exc)
    return map_ws_transport_error(
        exc,
        f"STT request timed out after {timeout_s}s",
    )


class _STTProviderProtocol(Protocol):
    """Provider-specific wire behavior behind the shared STT lifecycle."""

    provider: str
    protocol_name: str
    default_api_key_env: str
    requires_api_key: bool
    clean_close_is_terminal: bool

    def __init__(self, config: STTClientConfig, api_key: str | None) -> None: ...

    def build_ws_url(self, api_base: str) -> str: ...

    def headers(self) -> dict[str, str]: ...

    async def open_session(self, websocket: Any, model: str) -> None: ...

    def encode_chunk(
        self, chunk: bytes | memoryview, *, final: bool
    ) -> str | bytes: ...

    # Sequence, not list: list is invariant, so a provider returning
    # list[str] would not satisfy list[str | bytes]. The caller only
    # iterates the result, so covariance is what we actually want.
    def finish_messages(self) -> Sequence[str | bytes]: ...

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]: ...


def _stt_ws_url(
    api_base: str, path: str, query: dict[str, str | int | bool] | None = None
) -> str:
    normalized = api_base.rstrip("/") + "/"
    url = to_websocket_url(urljoin(normalized, path.lstrip("/")))
    return f"{url}?{urlencode(query)}" if query else url


class _OpenAIRealtimeSTTProtocol:
    """Shared OpenAI-style base64 PCM append framing."""

    default_api_key_env = "OPENAI_API_KEY"
    requires_api_key = False
    clean_close_is_terminal = False
    ws_path = ""

    def __init__(self, config: "STTClientConfig", api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key

    def build_ws_url(self, api_base: str) -> str:
        return _stt_ws_url(api_base, self.ws_path)

    def headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def encode_chunk(self, chunk: bytes | memoryview, *, final: bool = False) -> str:
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }
        )


class _VllmRealtimeProtocol(_OpenAIRealtimeSTTProtocol):
    """vllm_realtime: WebSocket /v1/realtime — base64 PCM16 chunks, deltas.

    Server -> Client: ``{"type": "session.created"}`` then
    ``transcription.delta`` / ``transcription.done`` / ``error``.
    """

    provider = "vllm"
    protocol_name = "v1_realtime_transcription"
    ws_path = "/v1/realtime"

    async def open_session(self, websocket: Any, model: str) -> None:
        msg = _decode_stt_json_message(await websocket.recv(), provider=self.provider)
        if msg.get("type") != "session.created":
            raise _STTProtocolError(f"Expected session.created, got: {msg}")
        await websocket.send(json.dumps({"type": "session.update", "model": model}))
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "input_audio_buffer.commit", "final": True})]

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "transcription.delta":
            return "delta", msg.get("delta", "")
        if msg_type == "transcription.done":
            return "done", msg.get("text", "")
        if msg_type == "error":
            return "error", str(msg.get("error", ""))
        return "", ""


class _VajraOpenAIRealtimeProtocol(_OpenAIRealtimeSTTProtocol):
    """vajra_openai_realtime: WebSocket /openai/v1/realtime — OpenAI transcription.

    Drives Vajra's OpenAI-compatible realtime transcription endpoint. Server ->
    Client: ``transcription_session.created`` then
    ``conversation.item.input_audio_transcription.delta`` /
    ``conversation.item.input_audio_transcription.completed`` / ``error``. v1 is
    manual-commit, one transcript per connection (PCM16 mono 16 kHz).
    """

    provider = "vajra"
    protocol_name = "openai_v1_realtime_transcription"
    ws_path = "/openai/v1/realtime?intent=transcription"

    async def open_session(self, websocket: Any, model: str) -> None:
        msg = _decode_stt_json_message(await websocket.recv(), provider=self.provider)
        if msg.get("type") != "transcription_session.created":
            raise _STTProtocolError(
                f"Expected transcription_session.created, got: {msg}"
            )
        # Configure the session; unlike vllm_realtime, do NOT commit here — a
        # commit before any audio would finalize an empty transcript.
        await websocket.send(
            json.dumps(
                {
                    "type": "transcription_session.update",
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {"model": model},
                }
            )
        )

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "input_audio_buffer.commit"})]

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "conversation.item.input_audio_transcription.delta":
            return "delta", msg.get("delta", "")
        if msg_type == "conversation.item.input_audio_transcription.completed":
            return "done", msg.get("transcript", "")
        # ".failed" is the terminal event for a committed item whose
        # transcription failed; treat it like a session-level "error" so the
        # request fails fast instead of waiting for the timeout.
        if msg_type in (
            "error",
            "conversation.item.input_audio_transcription.failed",
        ):
            error = msg.get("error")
            if isinstance(error, dict):
                return "error", str(error.get("message", ""))
            return "error", str(error or "")
        return "", ""


class _TogetherSTTProtocol(_OpenAIRealtimeSTTProtocol):
    """Together realtime transcription with explicit client-side endpointing.

    Together's interim transcription events replace the previous interim text,
    so they are normalized as snapshots. Disabling server VAD ensures that the
    single completion event corresponds to the explicit commit sent at EOF.
    """

    provider = "together"
    protocol_name = "together_openai_v1_realtime_transcription"
    default_api_key_env = "TOGETHER_API_KEY"
    requires_api_key = True

    def build_ws_url(self, api_base: str) -> str:
        return _stt_ws_url(
            api_base,
            "v1/realtime",
            {
                "intent": "transcription",
                "model": self.config.model,
                "input_audio_format": "pcm_s16le_16000",
                "language": self.config.language,
                "turn_detection": "none",
            },
        )

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

    async def open_session(self, websocket: Any, model: str) -> None:
        del model  # Model is selected in the WebSocket query string.
        msg = _decode_stt_json_message(await websocket.recv(), provider=self.provider)
        if msg.get("type") != "session.created":
            raise _STTProtocolError(f"Expected session.created, got: {msg}")

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "input_audio_buffer.commit"})]

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "conversation.item.input_audio_transcription.delta":
            return "snapshot", str(msg.get("delta") or "")
        if msg_type == "conversation.item.input_audio_transcription.completed":
            return "done", str(msg.get("transcript") or "")
        if msg_type in (
            "error",
            "conversation.item.input_audio_transcription.failed",
        ):
            error = msg.get("error")
            if isinstance(error, dict):
                return "error", str(error.get("message") or error)
            return "error", str(error or msg)
        return "", ""


class _DeepgramSTTProtocol:
    """Shared raw-PCM framing and authentication for Deepgram Listen APIs."""

    provider = "deepgram"
    default_api_key_env = "DEEPGRAM_API_KEY"
    requires_api_key = True
    clean_close_is_terminal = False
    endpoint = ""

    def __init__(self, config: "STTClientConfig", api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    async def open_session(self, websocket: Any, model: str) -> None:
        # Successful WebSocket upgrade means both Deepgram Listen APIs are
        # ready; neither requires a client-side session initialization frame.
        return None

    def encode_chunk(self, chunk: bytes | memoryview, *, final: bool = False) -> bytes:
        return bytes(chunk)

    def finish_messages(self) -> list[str]:
        return [json.dumps({"type": "CloseStream"})]

    @staticmethod
    def _error(msg: dict[str, Any]) -> tuple[str, str]:
        return (
            "error",
            str(msg.get("description") or msg.get("message") or msg.get("code") or msg),
        )


class _DeepgramNovaProtocol(_DeepgramSTTProtocol):
    protocol_name = "deepgram_v1_listen"
    endpoint = "v1/listen"

    def build_ws_url(self, api_base: str) -> str:
        return _stt_ws_url(
            api_base,
            self.endpoint,
            {
                "model": self.config.model,
                "encoding": "linear16",
                "sample_rate": self.config.sample_rate,
                "channels": 1,
                "language": self.config.language,
                "interim_results": "true",
                "punctuate": "true",
                "smart_format": "false",
                "mip_opt_out": str(self.config.mip_opt_out).lower(),
            },
        )

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "Results":
            channel = msg.get("channel")
            alternatives = (
                channel.get("alternatives") if isinstance(channel, dict) else None
            )
            first = (
                alternatives[0]
                if isinstance(alternatives, list) and alternatives
                else {}
            )
            text = first.get("transcript", "") if isinstance(first, dict) else ""
            return ("commit" if msg.get("is_final") else "snapshot"), str(text)
        if msg_type == "Metadata":
            return "done", ""
        if msg_type in ("Error", "error"):
            return self._error(msg)
        return "", ""


class _DeepgramFluxProtocol(_DeepgramSTTProtocol):
    protocol_name = "deepgram_v2_flux_listen"
    endpoint = "v2/listen"
    clean_close_is_terminal = True

    def build_ws_url(self, api_base: str) -> str:
        return _stt_ws_url(
            api_base,
            self.endpoint,
            {
                "model": self.config.model,
                "encoding": "linear16",
                "sample_rate": self.config.sample_rate,
                "mip_opt_out": str(self.config.mip_opt_out).lower(),
            },
        )

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "TurnInfo":
            kind = "commit" if msg.get("event") == "EndOfTurn" else "snapshot"
            return kind, str(msg.get("transcript") or "")
        if msg_type == "Metadata":
            return "done", ""
        if msg_type in ("Error", "error"):
            return self._error(msg)
        return "", ""


class _ElevenLabsSTTProtocol:
    provider = "elevenlabs"
    protocol_name = "elevenlabs_scribe_v2_realtime"
    default_api_key_env = "ELEVENLABS_API_KEY"
    requires_api_key = True
    clean_close_is_terminal = False

    def __init__(self, config: "STTClientConfig", api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_ws_url(self, api_base: str) -> str:
        return _stt_ws_url(
            api_base,
            "v1/speech-to-text/realtime",
            {
                "model_id": self.config.model,
                "audio_format": f"pcm_{self.config.sample_rate}",
                "language_code": self.config.language,
                "commit_strategy": "manual",
                "include_timestamps": "false",
            },
        )

    def headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    async def open_session(self, websocket: Any, model: str) -> None:
        msg = _decode_stt_json_message(await websocket.recv(), provider=self.provider)
        if msg.get("message_type") != "session_started":
            raise _STTProtocolError(f"Expected session_started, got: {msg}")

    def encode_chunk(self, chunk: bytes | memoryview, *, final: bool = False) -> str:
        return json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(chunk).decode("ascii"),
                "sample_rate": self.config.sample_rate,
                "commit": final,
            }
        )

    def finish_messages(self) -> list[str]:
        # The final audio chunk carries commit=true, so no second sentinel is
        # needed and no empty audio message can distort provider behavior.
        return []

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("message_type")
        if msg_type == "partial_transcript":
            return "snapshot", str(msg.get("text") or "")
        if msg_type == "committed_transcript":
            return "done", str(msg.get("text") or "")
        if isinstance(msg_type, str) and (
            msg_type.endswith("_error") or msg_type in {"error", "auth_error"}
        ):
            return "error", str(msg.get("message") or msg.get("error") or msg)
        return "", ""


class _MistralSTTProtocol:
    """Mistral realtime transcription WebSocket protocol."""

    provider = "mistral"
    protocol_name = "mistral_realtime_transcription"
    default_api_key_env = "MISTRAL_API_KEY"
    requires_api_key = True
    clean_close_is_terminal = False

    def __init__(self, config: "STTClientConfig", api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_ws_url(self, api_base: str) -> str:
        return _stt_ws_url(
            api_base,
            "v1/audio/transcriptions/realtime",
            {"model": self.config.model},
        )

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def open_session(self, websocket: Any, model: str) -> None:
        del model  # Model is selected in the WebSocket query string.
        msg = _decode_stt_json_message(await websocket.recv(), provider=self.provider)
        if msg.get("type") != "session.created":
            raise _STTProtocolError(f"Expected session.created, got: {msg}")
        session: dict[str, Any] = {
            "audio_format": {
                "encoding": "pcm_s16le",
                "sample_rate": self.config.sample_rate,
            }
        }
        if self.config.target_streaming_delay_ms is not None:
            session["target_streaming_delay_ms"] = self.config.target_streaming_delay_ms
        await websocket.send(json.dumps({"type": "session.update", "session": session}))

    def encode_chunk(self, chunk: bytes | memoryview, *, final: bool = False) -> str:
        del final
        return json.dumps(
            {
                "type": "input_audio.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
        )

    def finish_messages(self) -> list[str]:
        return [
            json.dumps({"type": "input_audio.flush"}),
            json.dumps({"type": "input_audio.end"}),
        ]

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "transcription.text.delta":
            return "delta", str(msg.get("text") or "")
        if msg_type == "transcription.done":
            return "done", str(msg.get("text") or "")
        if msg_type == "error":
            error = msg.get("error")
            if isinstance(error, dict):
                return "error", str(error.get("message") or error)
            return "error", str(error or msg)
        return "", ""


class _CartesiaSTTProtocol:
    """Cartesia manual realtime STT protocol with explicit finalization."""

    provider = "cartesia"
    protocol_name = "cartesia_stt_websocket_manual"
    default_api_key_env = "CARTESIA_API_KEY"
    requires_api_key = True
    clean_close_is_terminal = False

    def __init__(self, config: "STTClientConfig", api_key: str | None) -> None:
        self.config = config
        self.api_key = api_key or ""

    def build_ws_url(self, api_base: str) -> str:
        return _stt_ws_url(
            api_base,
            "stt/websocket",
            {
                "model": self.config.model,
                "encoding": "pcm_s16le",
                "sample_rate": self.config.sample_rate,
                "language": self.config.language,
            },
        )

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": self.config.cartesia_version,
        }

    async def open_session(self, websocket: Any, model: str) -> None:
        del websocket, model

    def encode_chunk(self, chunk: bytes | memoryview, *, final: bool = False) -> bytes:
        del final
        return bytes(chunk)

    def finish_messages(self) -> list[str]:
        return ["finalize", "close"]

    def parse_message(self, msg: dict[str, Any]) -> tuple[str, str]:
        msg_type = msg.get("type")
        if msg_type == "transcript":
            # Cartesia sends both provisional and final transcript payloads as
            # deltas, so their leading/trailing whitespace must survive until
            # the stream assembler concatenates them.
            kind = "final_delta" if msg.get("is_final") else "partial_delta"
            return kind, str(msg.get("text") or "")
        if msg_type == "done":
            return "done", ""
        if msg_type == "error":
            return "error", str(msg.get("message") or msg.get("title") or msg)
        return "", ""


_STT_PROTOCOLS: dict[str, type[_STTProviderProtocol]] = {
    "vllm_realtime": _VllmRealtimeProtocol,
    "vajra_openai_realtime": _VajraOpenAIRealtimeProtocol,
    "deepgram_flux": _DeepgramFluxProtocol,
    "deepgram_nova": _DeepgramNovaProtocol,
    "elevenlabs": _ElevenLabsSTTProtocol,
    "mistral": _MistralSTTProtocol,
    "cartesia": _CartesiaSTTProtocol,
    "together": _TogetherSTTProtocol,
}
