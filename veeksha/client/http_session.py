"""Shared aiohttp session construction for the HTTP clients.

Sessions bind to the event loop running when they are built, so each client
keeps its own thread-local session; this module only holds the transport
configuration they share.
"""

from __future__ import annotations

import threading

import aiohttp

# aiohttp cannot express "no keepalive expiry" (``keepalive_timeout=None``
# breaks its pool cleanup arithmetic), so use a value longer than any run.
KEEPALIVE_TIMEOUT_S = 24 * 60 * 60


def new_session(timeout_s: float) -> aiohttp.ClientSession:
    """Build a session whose timeouts apply per operation, not per request."""
    return aiohttp.ClientSession(
        # ``total`` stays unset so a long generation is not cut off mid-stream;
        # ``sock_read`` bounds the gap between chunks.
        timeout=aiohttp.ClientTimeout(
            total=None,
            connect=timeout_s,
            sock_connect=timeout_s,
            sock_read=timeout_s,
        ),
        connector=aiohttp.TCPConnector(
            # 0 means unlimited; aiohttp otherwise caps at 100 connections,
            # which would throttle a high-concurrency benchmark.
            limit=0,
            limit_per_host=0,
            keepalive_timeout=KEEPALIVE_TIMEOUT_S,
        ),
    )


async def close_session(storage: threading.local) -> None:
    """Close this thread's session, if one was ever built on it."""
    session = getattr(storage, "client", None)
    if session is not None:
        await session.close()
        del storage.client
