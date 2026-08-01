"""Shared helpers for OpenAI-compatible HTTP clients."""

from __future__ import annotations

import functools
import threading
from typing import Any, Dict

import httpx

from veeksha.client.base import BaseLLMClient
from veeksha.core.request import Request
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.types import ChannelModality


class OpenAIBaseClient(BaseLLMClient):
    """Common base for OpenAI-compatible clients.

    This includes:
    - Thread-local httpx.AsyncClient management
    - Token counting helpers
    - Global + per-request sampling params merge logic
    """

    def __init__(self, config, tokenizer_provider: TokenizerProvider) -> None:
        super().__init__(config)
        self.tokenizer_provider = tokenizer_provider
        self.client_storage = threading.local()

        # At this point either self.api_base and self.api_key are set or an error is raised.
        if not str(self.api_base).endswith("/"):
            self.api_base = str(self.api_base) + "/"

        self.text_tokenizer_handle = self.tokenizer_provider.for_modality(
            ChannelModality.TEXT
        )

    @functools.lru_cache(maxsize=10000)
    def _get_cached_token_count(self, text: str) -> int:
        """Return token count for text with caching."""
        return len(self.text_tokenizer_handle.encode(text))

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create a thread-local httpx client."""
        if not hasattr(self.client_storage, "client"):
            self.client_storage.client = httpx.AsyncClient(
                timeout=self.config.request_timeout
            )
        return self.client_storage.client

    def _get_sampling_params(self, request: Request) -> Dict[str, Any]:
        """Merge global and per-request sampling params.

        Per-request params live under ``request.metadata["sampling_params"]``.
        This also maps lm-eval convenience keys:
        - until -> stop
        - max_gen_toks -> config.max_tokens_param (if not already set)
        """
        params: Dict[str, Any] = {}
        global_params = getattr(self.config, "additional_sampling_params_dict", None)
        if isinstance(global_params, dict):
            params.update(global_params)

        if isinstance(request.metadata, dict):
            per_request = request.metadata.get("sampling_params")
            if isinstance(per_request, dict):
                params.update(per_request)

        # lm-eval generation kwargs
        if "until" in params:
            if "stop" not in params:
                params["stop"] = params["until"]
            params.pop("until", None)

        if "max_gen_toks" in params:
            max_gen_toks = params.pop("max_gen_toks", None)
            max_tokens_param = getattr(self.config, "max_tokens_param", None)
            if (
                max_tokens_param
                and max_gen_toks is not None
                and max_tokens_param not in params
            ):
                params[max_tokens_param] = max_gen_toks

        return params
