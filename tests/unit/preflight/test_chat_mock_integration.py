"""End-to-end: real chat client -> spawned mock server -> scorer.

Exercises the actual OpenAIChatCompletionsClient streaming path against a
deterministic mock server running in a separate process, then scores the
paired client/server timestamps. This is the vertical slice's proof that
request/response delivery lag and server pacing fidelity are measured correctly.
"""

import asyncio

import pytest

from veeksha.client import ClientRegistry
from veeksha.config.client import OpenAIChatCompletionsClientConfig
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.tokenizer import TokenizerHandle, TokenizerProvider
from veeksha.preflight import scorer
from veeksha.preflight.spawn import spawn_mock_chat_server
from veeksha.types import ChannelModality

TTFC_MS = 80.0
TPOC_MS = 10.0
NUM_CHUNKS = 24


def _word_tokenizer_provider() -> TokenizerProvider:
    handle = TokenizerHandle(
        count_tokens=lambda text: len(str(text).split()),
        decode=lambda token_ids: "",
        encode=lambda text: [0] * len(str(text).split()),
    )
    return TokenizerProvider({ChannelModality.TEXT: handle})


def _make_request(rid: int) -> Request:
    return Request(
        id=rid,
        channels={
            ChannelModality.TEXT: TextChannelRequestContent(input_text=f"hello {rid}")
        },
    )


@pytest.mark.unit
def test_chat_client_against_mock_is_measured():
    with spawn_mock_chat_server(
        ttfc_ms=TTFC_MS, tpoc_ms=TPOC_MS, num_chunks=NUM_CHUNKS
    ) as server:
        config = OpenAIChatCompletionsClientConfig(
            api_base=server.api_base,
            api_key="preflight",
            model="dummy",
            record_preflight_timing=True,
        )
        client = ClientRegistry.get(
            config.get_type(),
            config=config,
            tokenizer_provider=_word_tokenizer_provider(),
        )

        async def _run():
            results = []
            for rid in range(6):
                results.append(
                    await client.send_request(request=_make_request(rid), session_id=0)
                )
            await client.aclose()
            return results

        results = asyncio.run(_run())
        server_records = server.fetch_records()

    # every request succeeded and streamed all chunks
    assert all(r.success for r in results), [r.error_msg for r in results]
    assert all(len(r.chunk_recv_times) == NUM_CHUNKS for r in results)
    assert len(server_records) == len(results)

    report = scorer.score(results, server_records, ttfc_ms=TTFC_MS, tpoc_ms=TPOC_MS)

    # all requests paired with a server record
    assert report.n_paired_requests == len(results)
    assert report.unpaired_fraction == 0.0

    # Sample counts and ordering, not magnitudes: how large the timings are
    # depends on how loaded the machine running the suite is.
    m = report.metrics
    rd = m[scorer.M_REQUEST_DELIVERY]
    resp = m[scorer.M_RESPONSE_DELIVERY]
    assert rd.count == len(results)
    assert resp.count == NUM_CHUNKS * len(results)
    # A send is stamped before the receipt it is paired with, so a negative
    # delivery means mispaired records or a bad clock source, never noise.
    assert rd.minimum >= 0
    assert resp.minimum >= 0

    assert m[scorer.M_SERVER_TTFC_ABS_ERR].count == len(results)
    assert m[scorer.M_SERVER_TPOC_ABS_ERR].count == (NUM_CHUNKS - 1) * len(results)
    assert m[scorer.M_CLIENT_TTFC].count == len(results)
    assert m[scorer.M_CLIENT_TPOC].count == (NUM_CHUNKS - 1) * len(results)
