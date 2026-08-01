"""Router client for OpenAI-compatible APIs.

This client supports *per-request* routing between:
- `/chat/completions` (streaming) for generation-style requests
- `/completions` (non-stream) for logprobs/loglikelihood-style requests

Routing is controlled by `request.metadata["api_mode"]`:
- "chat" (or unset) -> chat completions
- "completions" -> completions
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, Optional

from veeksha.client.base import BaseLLMClient
from veeksha.client.openai_chat import OpenAIChatCompletionsClient
from veeksha.client.openai_completions import OpenAICompletionsClient
from veeksha.core.request import Request
from veeksha.core.response import RequestResult
from veeksha.core.tokenizer import TokenizerProvider

if TYPE_CHECKING:
    from veeksha.config.client import OpenAIRouterClientConfig


class OpenAIRouterClient(BaseLLMClient):
    """Routes requests to chat or completions endpoints based on request metadata."""

    def __init__(
        self, config: OpenAIRouterClientConfig, tokenizer_provider: TokenizerProvider
    ) -> None:
        super().__init__(config)
        self._chat_client = OpenAIChatCompletionsClient(
            config=config, tokenizer_provider=tokenizer_provider
        )

        # Create a config copy for completions with the correct max_tokens_param.
        # The /completions endpoint uses different param names than /chat/completions.
        completions_config = copy.copy(config)
        completions_max_tokens = getattr(config, "completions_max_tokens_param", None)
        if completions_max_tokens:
            # Override max_tokens_param for the completions sub-client
            object.__setattr__(
                completions_config, "max_tokens_param", completions_max_tokens
            )

        self._completions_client = OpenAICompletionsClient(
            config=completions_config, tokenizer_provider=tokenizer_provider
        )

    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        if isinstance(request.metadata, dict) and request.metadata.get("api_mode") == (
            "completions"
        ):
            return await self._completions_client.send_request(
                request=request,
                session_id=session_id,
                session_total_requests=session_total_requests,
                on_request_sent=on_request_sent,
                on_request_dispatched=on_request_dispatched,
            )
        return await self._chat_client.send_request(
            request=request,
            session_id=session_id,
            session_total_requests=session_total_requests,
            on_request_sent=on_request_sent,
            on_request_dispatched=on_request_dispatched,
        )
