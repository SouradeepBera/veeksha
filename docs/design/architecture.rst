System Architecture
===================

This page describes Veeksha's internal architecture, including how components
interact and how requests flow through the system.


High-level components
---------------------

Veeksha is composed of several key components that work together:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Benchmark Runner                                │
    │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
    │  │ Session Generator│  │ Traffic Scheduler│  │     Evaluator    │       │
    │  │  - synthetic     │  │  - rate-based    │  │  - performance   │       │
    │  │  - trace         │  │  - concurrent    │  │  - accuracy      │       │
    │  │  - lmeval        │  │                  │  │                  │       │
    │  └────────┬─────────┘  └────────┬─────────┘  └────────▲─────────┘       │
    │           │                     │                     │                 │
    │           ▼                     ▼                     │                 │
    │  ┌──────────────────────────────────────────┐         │                 │
    │  │              Worker Pool                 │         │                 │
    │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │         │                 │
    │  │  │ Prefetch │ │ Dispatch │ │Completion│  │         │                 │
    │  │  │ Workers  │→│ Workers  │→│ Workers  │──┼─────────┘                 │
    │  │  └──────────┘ └────┬─────┘ └──────────┘  │                           │
    │  └────────────────────┼─────────────────────┘                           │
    │                       │                                                 │
    │                       ▼                                                 │
    │  ┌──────────────────────────────────────────┐                           │
    │  │             Client Runners               │                           │
    │  │  - Async HTTP clients (httpx)            │                           │
    │  │  - Streaming response handling           │                           │
    │  └──────────────────────────────────────────┘                           │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   LLM Inference API   │
                        │  (vLLM, SGLang, etc.) │
                        └───────────────────────┘


Component descriptions
----------------------

**Session Generator**
    Creates ``Session`` objects representing user conversations or agentic flows. Each session
    contains a graph of requests with dependencies. Three types are available:

    - ``synthetic``: Generates random content with configurable distributions
    - ``trace``: Replays recorded conversation traces
    - ``lmeval``: Generates evaluation prompts from lm-eval-harness tasks

**Traffic Scheduler**
    Controls when sessions and their requests are dispatched. Handles:

    - Inter-session timing (arrival rate or target concurrency)
    - Intra-session dependencies (waiting for parent requests to complete)
    - History population (adding prior turns to request context)

**Worker Pool**
    Thread-based workers that process requests through the pipeline:

    - **Prefetch Workers**: Pre-generate sessions to ensure work is always ready
    - **Dispatch Workers**: Wait for ready requests and send them to clients
    - **Completion Workers**: Process completed requests and trigger next steps

**Client Runners**
    Async HTTP clients that actually communicate with the LLM inference API.
    Handle streaming responses and capture detailed timing information.

**Evaluator**
    Consumes completed requests and computes metrics. Supports:

    - Performance metrics (TTFC, TBC, TPOT, throughput)
    - Accuracy evaluation (lm-eval integration)
    - SLO checking (latency percentile thresholds)


Request lifecycle
-----------------

Every request goes through these stages with precise timestamp capture:

.. code-block:: text

    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │ scheduler_ready │────▶│scheduler_dispatch│────▶│ client_pickup   │
    │      _at        │     │       _at        │     │      _at        │
    └─────────────────┘     └──────────────────┘     └─────────────────┘
           │                        │                        │
           │                        │                        │
           ▼                        ▼                        ▼
    Request dependencies     Dispatcher thread      Client runner picks
    satisfied; request       pops from ready        up and sends HTTP
    enters ready queue       queue and marks        request to server
                             dispatched

    ┌─────────────────┐     ┌─────────────────┐
    │client_completed │────▶│result_processed │
    │      _at        │     │      _at        │
    └─────────────────┘     └─────────────────┘
           │                        │
           │                        │
           ▼                        ▼
    Full response received;  Completion worker
    client records final     processes result,
    timing                   notifies scheduler

These timestamps enable computing:

- **Dispatch delay**: ``scheduler_dispatched_at - scheduler_ready_at``
- **Queue wait**: ``client_picked_up_at - scheduler_dispatched_at``
- **Processing delay**: ``result_processed_at - client_completed_at``


Threading model
---------------

Veeksha uses a multi-threaded architecture with configurable worker counts:

.. code-block:: yaml

    runtime:
      num_dispatcher_threads: 2   # Threads for dispatching requests
      num_completion_threads: 2   # Threads for processing completions
      num_client_threads: 3       # Async worker threads for HTTP clients

**Dispatcher Threads**
    Wait on the traffic scheduler's ready queue and dispatch requests to
    client runners. More threads help when dispatch overhead is significant.

**Completion Threads**
    Process completed requests: update session state, notify the scheduler,
    and feed results to the evaluator. More threads help with high throughput.

**Client Threads**
    Each runs an async event loop with an ``httpx.AsyncClient`` for making
    concurrent HTTP requests. More threads increase I/O parallelism.

.. note::

    For optimal performance with free-threaded Python (3.14t), the GIL is
    disabled, allowing true parallelism across all worker threads.


Output pipeline
---------------

During and after the benchmark, several output mechanisms record data:

**Trace Recorder**
    Writes dispatched requests to ``traces/trace.jsonl`` as they are sent.
    Includes session context and optionally full request content.

**Evaluator**
    Accumulates metrics in memory and writes final results to ``metrics/``:

    - ``request_level_metrics.jsonl``: Per-request detailed data
    - ``*.csv``: Percentile distributions for each metric
    - ``*.png``: Distribution plots
    - ``summary_stats.json``: Aggregate statistics
    - ``slo_results.json``: SLO compliance results

**Health Checker**
    Post-benchmark verification that validates:

    - Session dispatch rate matches configuration
    - Request dependencies were respected
    - Prompt/output lengths match targets
    - Lifecycle timing is reasonable
