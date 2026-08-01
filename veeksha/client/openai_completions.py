"""Client for OpenAI-compatible Completions API."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import httpx

from veeksha.client.openai_base import OpenAIBaseClient
from veeksha.core.request import Request
from veeksha.core.response import ChannelResponse, RequestResult
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.logger import init_logger
from veeksha.types import ChannelModality

if TYPE_CHECKING:
    from veeksha.config.client import BaseClientConfig

logger = init_logger(__name__)


class OpenAICompletionsClient(OpenAIBaseClient):
    """Async client for OpenAI `/completions` API using httpx.

    This client always uses the completions endpoint (non-streaming). It is
    intended for servers that only expose completions, or workloads that require
    provider logprobs (e.g., lm-eval loglikelihood requests).
    """

    def __init__(
        self,
        config: BaseClientConfig,
        tokenizer_provider: TokenizerProvider,
    ) -> None:
        super().__init__(config=config, tokenizer_provider=tokenizer_provider)

        append = str(self.config.address_append_value)
        if append.endswith("chat/completions"):
            completions_path = append.replace("chat/completions", "completions")
        else:
            completions_path = "completions"
        self.completions_address = str(self.api_base) + completions_path

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        """Send a request to the OpenAI `/completions` endpoint (non-streaming)."""
        return await self._send_completions_request(
            request=request,
            session_id=session_id,
            session_total_requests=session_total_requests,
            on_request_sent=on_request_sent,
            on_request_dispatched=on_request_dispatched,
        )

    async def _send_completions_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        """Execute the HTTP request against `/completions` and parse the response."""
        timeout = self.config.request_timeout

        prompt_text = ""
        if ChannelModality.TEXT in request.channels:
            text_content = request.channels[ChannelModality.TEXT]
            prompt_text = text_content.input_text  # type: ignore

        max_tokens_limit = None
        if (
            request.requested_output is not None
            and request.requested_output.text is not None
        ):
            max_tokens_limit = request.requested_output.text.target_tokens

        sampling_params = self._get_sampling_params(request)
        body: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt_text,
            "stream": False,
        }
        body.update(sampling_params)
        body["stream"] = False

        # Max tokens handling:
        # - Prefer explicit per-request sampling params
        # - Otherwise use the configured max_tokens_param
        max_tokens_param = getattr(self.config, "max_tokens_param", None)
        has_explicit_max_tokens = "max_tokens" in body
        has_configured_max_tokens = bool(max_tokens_param and max_tokens_param in body)
        if (
            max_tokens_limit is not None
            and int(max_tokens_limit) > 0
            and max_tokens_param
            and not has_explicit_max_tokens
            and not has_configured_max_tokens
        ):
            body[max_tokens_param] = max_tokens_limit

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        error_msg: Optional[str] = None
        error_code: Optional[int] = None
        generated_text = ""
        completion_text = ""
        logprobs: Any = None

        # preflight timing (recorded only when enabled). Only the request id is
        # sent to the server; the scorer joins the two record books by request_id.
        # Completions is non-streaming: a single receipt stamp.
        preflight_enabled = getattr(self.config, "record_preflight_timing", False)
        client_sent_at: Optional[float] = None
        chunk_recv_times: List[float] = []

        start_time = time.monotonic()
        client_sent_at = start_time
        if preflight_enabled:
            headers["X-Veeksha-Request-Id"] = str(request.id)
        try:
            client = self._get_client()
            response = await client.post(
                self.completions_address, json=body, headers=headers, timeout=timeout
            )
            # Response fully received (non-streaming).
            receive_time = time.monotonic()
            response.raise_for_status()
            if on_request_dispatched is not None:
                on_request_dispatched()
            if on_request_sent is not None:
                on_request_sent()
            data = response.json()
            if preflight_enabled:
                chunk_recv_times.append(receive_time)
            choices = data.get("choices") or []
            if choices:
                first = choices[0] if isinstance(choices[0], dict) else {}
                generated_text = first.get("text", "") or ""
                logprobs = first.get("logprobs")
        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code if e.response else 500
            error_msg = error_msg or str(e)
            logger.warning("HTTP Error: status=%s msg=%s", error_code, error_msg)
        except httpx.ConnectError as e:
            error_code = 503
            error_msg = error_msg or str(e)
            logger.warning("Connection Error: (%s) %s", error_code, error_msg)
        except httpx.TimeoutException:
            error_code = 408
            error_msg = error_msg or "Request timed out"
            logger.warning("Timeout Error: (%s) %s", error_code, error_msg)
        except Exception as e:
            error_code = error_code or 520
            error_msg = error_msg or str(e)
            logger.exception("Unexpected error: (%s) %s", error_code, error_msg)

        completed_at = time.monotonic()
        success = error_msg is None and error_code is None

        inter_chunk_times: List[float] = []
        num_delta_prompt_tokens = 0
        num_total_prompt_tokens = 0
        num_output_tokens = 0
        if success:
            end_to_end = max(0.0, completed_at - start_time)
            inter_chunk_times = [end_to_end]
            num_total_prompt_tokens = len(
                self.text_tokenizer_handle.encode(prompt_text)
            )
            num_delta_prompt_tokens = num_total_prompt_tokens

            completion_text = generated_text
            if (
                body.get("echo")
                and prompt_text
                and generated_text.startswith(prompt_text)
            ):
                completion_text = generated_text[len(prompt_text) :]

            num_output_tokens = len(self.text_tokenizer_handle.encode(completion_text))
            if max_tokens_limit is not None:
                num_output_tokens = min(int(max_tokens_limit), num_output_tokens)

        channels: Dict[ChannelModality, ChannelResponse] = {}
        if success:
            channels[ChannelModality.TEXT] = ChannelResponse(
                modality=ChannelModality.TEXT,
                content=completion_text,
                metrics={
                    "is_stream": False,
                    "inter_chunk_times": inter_chunk_times,
                    "num_delta_prompt_tokens": num_delta_prompt_tokens,
                    "num_total_prompt_tokens": num_total_prompt_tokens,
                    "num_output_tokens": num_output_tokens,
                    "logprobs": logprobs,
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
