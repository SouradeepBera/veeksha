"""End-to-end: the preflight driver runs the REAL benchmark loop vs a mock.

Unlike test_chat_mock_integration (which drives the client directly), this
exercises the full scheduler -> dispatch -> client_runner -> client path, so the
dispatch-drift metrics are populated too. Then it gates the result.
"""

import pytest

from veeksha.config.preflight import (
    PreflightSttCheckConfig,
    PreflightTextCheckConfig,
    PreflightTtsCheckConfig,
)
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.traffic import ConcurrentTrafficConfig
from veeksha.preflight import scorer, validator
from veeksha.preflight.drivers import (
    run_completions_preflight,
    run_realtime_tts_preflight,
    run_stt_preflight,
    run_text_preflight,
    run_tts_preflight,
    run_vajra_tts_preflight,
)

TTFC_MS = 60.0
TPOC_MS = 8.0
NUM_CHUNKS = 16


def _traffic(concurrency):
    return ConcurrentTrafficConfig(
        target_concurrent_sessions=concurrency, rampup_seconds=0
    )


def _text_cfg():
    return PreflightTextCheckConfig(
        input_tokens=8,
        num_response_chunks=NUM_CHUNKS,
        server_ttfc_ms=TTFC_MS,
        server_tpoc_ms=TPOC_MS,
    )


def _tts_cfg():
    return PreflightTtsCheckConfig(
        input_tokens=8,
        input_chunk_tokens=2,
        input_pacing_tps=500.0,  # fast so the test isn't slow
        num_response_chunks=NUM_CHUNKS,
        server_ttfc_ms=TTFC_MS,
        server_tpoc_ms=TPOC_MS,
    )


def _stt_cfg():
    return PreflightSttCheckConfig(
        input_seconds=0.3,  # short clip keeps the realtime-paced input quick
        input_chunk_bytes=1024,
        sample_rate=16000,
        num_response_chunks=NUM_CHUNKS,
        server_ttfc_ms=TTFC_MS,
        server_tpoc_ms=TPOC_MS,
    )


def _assert_measured(report, streaming_input):
    assert report.n_requests > 0
    assert report.n_paired_requests > 0
    assert report.unpaired_fraction < 0.5
    m = report.metrics
    assert m[scorer.M_REQUEST_DELIVERY].count > 0
    assert m[scorer.M_RESPONSE_DELIVERY].count > 0
    assert m[scorer.M_SERVER_TTFC_ABS_ERR].count > 0
    # dispatch drift only exists because the real scheduler/dispatch path ran.
    assert m[scorer.M_LIFECYCLE_READY_TO_SEND].count > 0
    if streaming_input:
        assert m[scorer.M_INPUT_DELIVERY].count > 0
        assert m[scorer.M_INPUT_PACING_ABS_ERR].count > 0
    else:
        assert scorer.M_INPUT_DELIVERY not in m
        assert scorer.M_INPUT_PACING_ABS_ERR not in m


@pytest.mark.unit
def test_text_driver_runs_real_loop_and_scores(tmp_path):
    report = run_text_preflight(
        _text_cfg(),
        traffic_scheduler=_traffic(8),
        runtime=RuntimeConfig(max_sessions=40),
        output_dir=str(tmp_path / "text"),
    )
    _assert_measured(report, streaming_input=False)
    assert report.metrics[scorer.M_SERVER_TPOC_ABS_ERR].count > 0

    result = validator.run_validation(
        report,
        delivery_lag_threshold_ms=25.0,
        server_pacing_threshold_ms=25.0,
        dispatch_drift_threshold_ms=50.0,
        input_pacing_threshold_ms=50.0,
        max_unpaired_fraction=0.1,
    )
    assert result.verdict in (
        validator.VERDICT_PASS,
        validator.VERDICT_FAIL,
        validator.VERDICT_SERVER_AT_CAPACITY,
    )


@pytest.mark.unit
def test_completions_driver_runs_real_loop_and_scores(tmp_path):
    report = run_completions_preflight(
        _text_cfg(),
        traffic_scheduler=_traffic(8),
        runtime=RuntimeConfig(max_sessions=40),
        output_dir=str(tmp_path / "completions"),
    )
    _assert_measured(report, streaming_input=False)
    # non-streaming: a single response, so no tpoc samples.
    assert scorer.M_SERVER_TPOC_ABS_ERR not in report.metrics


@pytest.mark.unit
def test_tts_driver_runs_real_loop_and_scores(tmp_path):
    report = run_tts_preflight(
        _tts_cfg(),
        traffic_scheduler=_traffic(8),
        runtime=RuntimeConfig(max_sessions=40),
        output_dir=str(tmp_path / "tts"),
    )
    # HTTP tts sends the whole text in one POST -> no streaming-input metrics.
    _assert_measured(report, streaming_input=False)
    assert report.metrics[scorer.M_SERVER_TPOC_ABS_ERR].count > 0


@pytest.mark.unit
def test_realtime_tts_driver_runs_real_loop_and_scores(tmp_path):
    report = run_realtime_tts_preflight(
        _tts_cfg(),
        traffic_scheduler=_traffic(6),
        runtime=RuntimeConfig(max_sessions=24),
        output_dir=str(tmp_path / "realtime_tts"),
    )
    _assert_measured(report, streaming_input=True)
    assert report.metrics[scorer.M_SERVER_TPOC_ABS_ERR].count > 0


@pytest.mark.unit
def test_vajra_tts_driver_runs_real_loop_and_scores(tmp_path):
    report = run_vajra_tts_preflight(
        _tts_cfg(),
        traffic_scheduler=_traffic(6),
        runtime=RuntimeConfig(max_sessions=24),
        output_dir=str(tmp_path / "vajra_tts"),
    )
    _assert_measured(report, streaming_input=True)
    assert report.metrics[scorer.M_SERVER_TPOC_ABS_ERR].count > 0


@pytest.mark.unit
def test_stt_driver_runs_real_loop_and_scores(tmp_path):
    report = run_stt_preflight(
        _stt_cfg(),
        traffic_scheduler=_traffic(4),
        runtime=RuntimeConfig(max_sessions=16),
        output_dir=str(tmp_path / "stt"),
    )
    _assert_measured(report, streaming_input=True)
    assert report.metrics[scorer.M_SERVER_TPOC_ABS_ERR].count > 0
