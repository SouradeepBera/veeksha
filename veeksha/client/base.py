"""Base LLM client abstract class for the new Veeksha framework."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

from veeksha.core.request import Request
from veeksha.core.response import RequestResult

if TYPE_CHECKING:
    from veeksha.config.client import BaseClientConfig


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    Clients are responsible for sending requests to LLM APIs and returning
    structured results that evaluators can consume.
    """

    def __init__(self, config: BaseClientConfig) -> None:
        """Initialize the client.

        Args:
            config: Client configuration with model, timeout, etc.
        """
        self.config = config
        self.api_base = config.api_base
        self.api_key = config.api_key

        api_base_env = os.environ.get("OPENAI_API_BASE", None)
        api_key_env = os.environ.get("OPENAI_API_KEY", None)

        if self.api_base is None:
            if api_base_env is not None:
                self.api_base = api_base_env
            else:
                raise ValueError("API base is not set in neither config nor env.")

        if self.api_key is None:
            if api_key_env is not None:
                self.api_key = api_key_env
            # don't raise for empty key

    @property
    def model_name(self) -> str:
        """Model identifier for API calls."""
        return self.config.model

    @abstractmethod
    async def send_request(
        self,
        request: Request,
        session_id: int,
        session_total_requests: int = 1,
        on_request_sent: Optional[Callable[[], None]] = None,
        on_request_dispatched: Optional[Callable[[], None]] = None,
    ) -> RequestResult:
        """Send a request to the LLM API.

        Args:
            request: The request to send (with channels)
            session_id: Session this request belongs to
            session_total_requests: Total number of requests in this session
            on_request_sent: Optional callback invoked once the server
                acknowledges the request (HTTP 200 received).
            on_request_dispatched: Optional callback invoked once the HTTP
                request is sent and the server returns HTTP 200, before any
                response chunks are processed.

        Returns:
            RequestResult containing response data and timing
        """
        raise NotImplementedError
