# Client loop count: root cause of the aiohttp preflight regression

Companion to `AIOHTTP_FINDINGS.md`. Measured on this machine (256 cores,
free-threaded CPython 3.14t, aiohttp 3.14.3) with:

```
veeksha preflight --concurrency 512 --num_sessions 1500 \
  --check_completions false --check_tts false --check_realtime_tts false \
  --check_vajra_tts false --check_stt false
```

Phase numbers come from an aiohttp `TraceConfig` attached to the real session
(`veeksha/client/_trace.py`, enabled with `VEEKSHA_TRACE=1`; no-op otherwise).

## 1. How many client threads

**8–16.** Today it is 64 at concurrency 512, from
`num_client_threads = ceil(concurrency / 8)` (`benchmark.py:111`).

Pristine HEAD, only this varied, 3 reps each:

| loops | request_delivery p99 | response_delivery p99 | tpoc p50 |
|---|---|---|---|
| 64 (current) | 2574–2775 ms | 1121–1262 ms | 0.003 ms |
| 16 | 18–140 ms | 0.44–0.48 ms | 19.94–20.03 ms |
| 8 | 2.2 / 2.6 ms | 0.41 / 0.43 ms | 19.99 / 20.01 ms |

16 is still clean, so cap the heuristic rather than pinning it to 8.

## 2. Why so, with aiohttp

The loops **block**, and that blocking scales with loop count — not with
concurrency, and not with per-loop work.

- Client threads are **0.4% CPU busy** (9.8 s CPU against 2752 thread-seconds)
  while carrying **1500–2600 s of aggregate loop lag**. Not CPU starvation:
  contention.
- TraceConfig puts all of it in connect:

  | phase | p50 |
  |---|---|
  | `connect` | 1131.6 ms |
  | ├─ `connect.resolve` | 0.008 ms (literal-IP path, no DNS) |
  | ├─ `connect.sock_connect` | 631.5 ms |
  | └─ `connect.transport_setup` | 435.4 ms |
  | `acquire_to_headers_sent` | 9.1 ms |

  `connect` p50 is essentially all of `request_delivery` p50. Connect is not
  itself broken — `transport_setup` is `loop.create_connection()` on an
  *already connected* socket and never enters the kernel, yet still costs
  435 ms. Connect simply needs the most loop turns, so it accumulates the most
  stall.
- One named offender: `ThreadPoolExecutor.submit` measured at **9 ms p50,
  called synchronously inside the loop thread** — the process-global
  `concurrent.futures.thread._global_shutdown_lock`, entered from the per-poll
  `run_in_executor` at `client_runner.py:68`.

## 3. Why it didn't matter for httpx

**Caveat:** httpx was never re-run in the harness at 64 loops as part of this
work. This rests on the httpx numbers in `AIOHTTP_FINDINGS.md`, which are
single runs — and identical 64-loop aiohttp configs were measured spanning
**227–1722 ms** `request_delivery` p50. Treat this section as weaker evidence
than section 1.

What is structural and does discriminate: the stall concentrates in connect,
and aiohttp's connect needs strictly more loop round-trips than httpx's.
`aiohappyeyeballs.start_connection` builds a raw socket and awaits
`loop.sock_connect`, *then* hands the socket to
`loop.create_connection(sock=...)`; httpx/anyio issues one `create_connection`.
Because the mock replies `Connection: close`, every request pays it —
**1500 connections created, 0 reused**. More loop turns per request, times 64
contending loops, means aiohttp hits the wall first.

The inverse framing in `AIOHTTP_FINDINGS.md` §2 ("httpx gains from
free-threading, aiohttp loses") does not hold. At 8 loops aiohttp beats the
httpx figures outright: response_delivery p99 **0.41 ms vs 0.671 ms**,
request_delivery p50 **0.81 ms vs 13 ms**.

## Corrections to `AIOHTTP_FINDINGS.md`

- **§4** — the `cpu/wall` "effective parallelism" metric is misleading. httpx
  scores 3.0× largely because it burns ~3× the CPU per request, not because it
  is better parallelized. Direct `time.thread_time()` on the worker threads is
  the load-bearing measurement, and it says the loops are idle-blocked.
- **§7** — both latent issues are real (the 9 ms `submit` is measured), but
  neither is required. Fixing both, plus `sock_read=None`, was
  neutral-to-slightly-worse once the loop count was right.
- Single-run comparisons throughout are inside the noise band quoted above.
- A GC hypothesis was chased and **rejected**: one `gc.disable()` run looked
  like a breakthrough (tpoc 0.003 → 18.96 ms) but did not survive two repeats.
