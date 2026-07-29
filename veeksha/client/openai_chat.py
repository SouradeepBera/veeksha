from __future__ import annotations

import asyncio
import base64
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import aiohttp

from veeksha.client.openai_base import OpenAIBaseClient
from veeksha.core.audio_contract import DEFAULT_AUDIO_SAMPLE_RATE, AudioMetricKey
from veeksha.core.request import Request
from veeksha.core.request_content import (
    AudioChannelRequestContent,
    ImageChannelRequestContent,
    TextChannelRequestContent,
    VideoChannelRequestContent,
)
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.logger import init_logger
from veeksha.types import AudioTask, ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import OpenAIChatCompletionsClientConfig

logger = init_logger(__name__)


class OpenAIChatCompletionsClient(OpenAIBaseClient):
    """Async client for OpenAI Chat Completions API using aiohttp.

    Works with new Request objects that have channels instead of prompt tuples.
    """

    def __init__(
        self,
        config: OpenAIChatCompletionsClientConfig,
        tokenizer_provider: TokenizerProvider,
    ) -> None:
        """Initialize the OpenAI Chat client.

        Args:
            config: Client configuration with model, timeout, etc.
            tokenizer_provider: Provider for tokenizers per modality.
        """
        super().__init__(config=config, tokenizer_provider=tokenizer_provider)
        self.chat_address = str(self.api_base) + str(self.config.address_append_value)

    def _build_text_content_block(
        self, text_content: TextChannelRequestContent
    ) -> tuple[dict, int]:
        """Build a text content block for multimodal messages.

        Args:
            text_content: Text content from request channels.

        Returns:
            Tuple of (content_block_dict, token_count).
        """
        prompt_text = text_content.input_text
        prompt_len = len(self.text_tokenizer_handle.encode(prompt_text))

        return {"type": "text", "text": prompt_text}, prompt_len

    def _build_image_content_block(
        self, image_content: ImageChannelRequestContent
    ) -> dict:
        """Build an image content block for multimodal messages."""
        return {"type": "image_url", "image_url": {"url": image_content.input_image}}

    def _build_audio_content_block(
        self, audio_content: AudioChannelRequestContent
    ) -> dict:
        """Build an audio content block for multimodal messages."""
        return {"type": "audio_url", "audio_url": {"url": audio_content.input_audio}}

    def _build_video_content_block(
        self, video_content: VideoChannelRequestContent
    ) -> dict:
        """Build a video content block for multimodal messages."""
        return {"type": "video_url", "video_url": {"url": video_content.input_video}}

    def _build_message_content(self, request: Request) -> tuple[list, int]:
        """Build multimodal message content from request channels.

        Constructs a list of content blocks in OpenAI multimodal format.

        Args:
            request: Request with channels dict mapping modalities to content.

        Returns:
            Tuple of (content_blocks_list, text_token_count).
        """
        content_blocks: List[dict] = []
        text_token_count = 0
        if ChannelModality.TEXT in request.channels:
            text_block, text_token_count = self._build_text_content_block(
                request.channels[ChannelModality.TEXT]  # type: ignore
            )
            content_blocks.append(text_block)

        if ChannelModality.IMAGE in request.channels:
            image_block = self._build_image_content_block(
                request.channels[ChannelModality.IMAGE]  # type: ignore
            )
            content_blocks.append(image_block)

        if ChannelModality.AUDIO in request.channels:
            audio_block = self._build_audio_content_block(
                request.channels[ChannelModality.AUDIO]  # type: ignore
            )
            content_blocks.append(audio_block)

        if ChannelModality.VIDEO in request.channels:
            video_block = self._build_video_content_block(
                request.channels[ChannelModality.VIDEO]  # type: ignore
            )
            content_blocks.append(video_block)

        messages = []
        if request.history:
            messages.extend(request.history)

        # current request content
        if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
            messages.append({"role": "user", "content": content_blocks[0]["text"]})
        else:
            messages.append({"role": "user", "content": content_blocks})

        return messages, text_token_count

    # ---------- response processing

    def _process_text_delta(
        self,
        delta_content: str,
        receive_time: float,
        most_recent_token_time: float,
        inter_chunk_times: List[float],
        generated_text: str,
        chunks_received: int,
    ) -> tuple[str, int, float, List[float]]:
        """Process a text delta from the streaming response.

        Args:
            delta_content: Text content from the delta.
            receive_time: Time of chunk receipt.
            most_recent_token_time: Time of last token received.
            inter_chunk_times: List of inter-chunk times.
            generated_text: Accumulated generated text.
            chunks_received: Total chunks received so far.

        Returns:
            Tuple of (generated_text, chunks_received,
                     most_recent_token_time, inter_chunk_times).
        """
        # treat each streaming delta as a single token
        generated_text += delta_content
        delta_duration = receive_time - most_recent_token_time
        inter_chunk_times.append(delta_duration)
        chunks_received += 1
        most_recent_token_time = receive_time

        return (
            generated_text,
            chunks_received,
            most_recent_token_time,
            inter_chunk_times,
        )

    def _process_image_response(
        self,
        delta: dict,
        image_data: Optional[Any],
    ) -> Optional[Any]:
        """Process image data from the streaming response.

        Args:
            delta: Delta dict from the streaming response.
            image_data: Accumulated image data (if any).

        Returns:
            Updated image data or None.
        """
        # Skeleton: Log warning and return None
        # Future: Parse image data from delta and accumulate
        return None

    def _process_audio_response(
        self,
        delta: dict,
        audio_data: Optional[Any],
    ) -> Optional[Any]:
        """Process audio data from the streaming response.

        Args:
            delta: Delta dict from the streaming response.
            audio_data: Accumulated audio data (if any).

        Returns:
            Updated audio data or None.
        """
        # Skeleton: Log warning and return None
        # Future: Parse audio chunks from delta and accumulate
        return None

    def _process_video_response(
        self,
        delta: dict,
        video_data: Optional[Any],
    ) -> Optional[Any]:
        """Process video data from the streaming response.

        Args:
            delta: Delta dict from the streaming response.
            video_data: Accumulated video data (if any).

        Returns:
            Updated video data or None.
        """
        # Skeleton: Log warning and return None
        # Future: Parse video data from delta and accumulate
        return None

    def _build_channel_responses(
        self,
        success: bool,
        generated_text: str,
        inter_chunk_times: List[float],
        delta_prompt_len: int,
        total_prompt_len: int,
        tokens_received: int,
        image_data: Optional[Any],
        audio_data: Optional[Any],
        video_data: Optional[Any],
    ) -> dict:
        """Build channel responses for all modalities.

        Args:
            success: Whether the request was successful.
            generated_text: Generated text output.
            inter_chunk_times: List of inter-chunk times.
            delta_prompt_len: Number of prompt tokens in the delta.
            total_prompt_len: Number of prompt tokens in the total prompt.
            tokens_received: Number of output tokens.
            image_data: Image response data (if any).
            audio_data: Audio response data (if any).
            video_data: Video response data (if any).

        Returns:
            Dict mapping ChannelModality to ChannelResponse.
        """
        channels = {}

        if not success:
            return channels

        if generated_text:
            channels[ChannelModality.TEXT] = ChannelResponse(
                modality=ChannelModality.TEXT,
                content=generated_text,
                metrics={
                    "is_stream": True,
                    "inter_chunk_times": inter_chunk_times,
                    "num_delta_prompt_tokens": delta_prompt_len,
                    "num_total_prompt_tokens": total_prompt_len,
                    "num_output_tokens": tokens_received,
                },
            )

        if image_data is not None:
            channels[ChannelModality.IMAGE] = ChannelResponse(
                modality=ChannelModality.IMAGE,
                content=image_data,
                metrics={},
            )

        if audio_data is not None:
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_data,
                metrics={"audio_task": AudioTask.LLM_AUDIO},
            )

        if video_data is not None:
            channels[ChannelModality.VIDEO] = ChannelResponse(
                modality=ChannelModality.VIDEO,
                content=video_data,
                metrics={},
            )

        return channels

    async def _process_stream(self, response: aiohttp.ClientResponse):
        """Process SSE stream from server."""
        import json

        buffer = b""
        async for chunk in response.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith(b"data:"):
                    data_str = line[5:].strip()
                    if data_str == b"[DONE]":
                        return
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        """Send a streaming request to the OpenAI Chat Completions API."""

        max_tokens_limit = None
        if (
            request.requested_output is not None
            and request.requested_output.text is not None
        ):
            max_tokens_limit = request.requested_output.text.target_tokens

        # text metrics
        inter_chunk_times: List[float] = []
        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        chunks_received = 0
        generated_text = ""

        # preflight timing (recorded only when enabled). Only the request id is
        # sent to the server; the scorer joins the two record books by request_id.
        preflight_enabled = getattr(self.config, "record_preflight_timing", False)
        client_sent_at: Optional[float] = None
        chunk_recv_times: List[float] = []

        # multimodal response data
        image_data: Optional[Any] = None
        video_data: Optional[Any] = None

        # audio output tracking
        audio_chunks: List[bytes] = []
        audio_ttfc: Optional[float] = None
        audio_chunk_count = 0

        delta_prompt_len = 0
        messages = []
        t_start = time.monotonic()

        try:
            messages, delta_prompt_len = self._build_message_content(request)

            body = {
                "model": self.config.model,
                "messages": messages,
                "stream": True,
                "ignore_eos": self.config.ignore_eos,  # type: ignore
            }
            body.update(self._get_sampling_params(request))

            max_tokens_param = self.config.max_tokens_param  # type: ignore
            if (
                max_tokens_limit is not None
                and int(max_tokens_limit) > 0
                and max_tokens_param
                and max_tokens_param
                not in body  # don't override the one set by the request
            ):
                body[max_tokens_param] = max_tokens_limit

            min_tokens_param = self.config.min_tokens_param  # type: ignore
            if (
                min_tokens_param
                and max_tokens_limit is not None
                and int(max_tokens_limit) > 0
                and min_tokens_param
                not in body  # don't override the one set by the request
            ):
                body[min_tokens_param] = max_tokens_limit

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

            client = self._get_client()
            t_start = time.monotonic()
            most_recent_token_time = t_start
            # t_cs: request handed to the transport (always recorded).
            client_sent_at = t_start
            if preflight_enabled:
                headers["X-Veeksha-Request-Id"] = str(request.id)
            async with client.post(
                self.chat_address,
                json=body,
                headers=headers,
            ) as response:
                response.raise_for_status()

                if on_request_dispatched is not None:
                    on_request_dispatched()

                sent_notified = False
                async for data in self._process_stream(response):
                    receive_time = time.monotonic()
                    # t_cr_i: client receipt of each response chunk.
                    if preflight_enabled:
                        chunk_recv_times.append(receive_time)
                    if "error" in data:
                        err = data.get("error") or {}
                        error_msg = err.get("message", "Unknown error")
                        code_value = err.get("code")
                        error_code = code_value if isinstance(code_value, int) else 400
                        break

                    choices = data.get("choices")
                    if not choices:
                        continue

                    first_choice = choices[0]
                    delta = first_choice.get("delta")
                    if not isinstance(delta, dict):
                        logger.warning(
                            "Streaming event missing delta: %s", first_choice
                        )
                        continue

                    modality_type = data.get("modality")

                    if modality_type == "audio" and delta.get("content"):
                        chunk_bytes = base64.b64decode(delta["content"])
                        if audio_ttfc is None:
                            audio_ttfc = (receive_time - t_start) * 1000
                        audio_chunks.append(chunk_bytes)
                        audio_chunk_count += 1
                    elif delta.get("content"):
                        (
                            generated_text,
                            chunks_received,
                            most_recent_token_time,
                            inter_chunk_times,
                        ) = self._process_text_delta(
                            delta["content"],
                            receive_time,
                            most_recent_token_time,
                            inter_chunk_times,
                            generated_text,
                            chunks_received,
                        )

                        # Notify after first content chunk (prefill done).
                        if not sent_notified and on_request_sent is not None:
                            on_request_sent()
                            sent_notified = True

                    # TODO: image deltas
                    image_data = self._process_image_response(delta, image_data)

                    # TODO: video deltas
                    video_data = self._process_video_response(delta, video_data)

        except aiohttp.ClientResponseError as e:
            error_code = e.status or 500
            error_msg = error_msg or str(e)
            logger.warning(f"HTTP Error: status={error_code} msg={error_msg}")
        except asyncio.TimeoutError:
            error_code = 408
            error_msg = error_msg or "Request timed out"
            logger.warning(f"Timeout Error: ({error_code}) {error_msg}")
        except aiohttp.ClientConnectorError as e:
            error_code = 503
            error_msg = error_msg or str(e)
            logger.warning(f"Connection Error: ({error_code}) {error_msg}")
        except Exception as e:
            error_code = error_code or 520
            error_msg = error_msg or str(e)
            logger.exception(f"Unexpected error: ({error_code}) {error_msg}")

        completed_at = time.monotonic()
        success = error_msg is None and error_code is None

        num_total_prompt_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                num_total_prompt_tokens += self._get_cached_token_count(content)
            elif isinstance(content, list):
                # multimodal format
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        num_total_prompt_tokens += self._get_cached_token_count(text)

        num_completion_tokens = len(self.text_tokenizer_handle.encode(generated_text))

        channels = self._build_channel_responses(
            success=success,
            generated_text=generated_text,
            inter_chunk_times=inter_chunk_times,
            delta_prompt_len=delta_prompt_len,
            total_prompt_len=num_total_prompt_tokens,
            tokens_received=num_completion_tokens,
            image_data=image_data,
            audio_data=None,
            video_data=video_data,
        )

        # Build audio channel from streamed audio chunks
        if success and audio_chunks:
            audio_bytes = b"".join(audio_chunks)
            total_latency_ms = (completed_at - t_start) * 1000
            channels[ChannelModality.AUDIO] = ChannelResponse(
                modality=ChannelModality.AUDIO,
                content=audio_bytes,
                metrics={
                    "audio_task": AudioTask.LLM_AUDIO,
                    AudioMetricKey.TTFC.value: round(audio_ttfc or 0.0, 3),
                    AudioMetricKey.END_TO_END_LATENCY.value: round(total_latency_ms, 3),
                    AudioMetricKey.CHUNK_COUNT.value: audio_chunk_count,
                    AudioMetricKey.RAW_PCM.value: True,
                    AudioMetricKey.SAMPLE_RATE.value: DEFAULT_AUDIO_SAMPLE_RATE,
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
