"""Unit tests for the preflight gate/verdict validator and report renderer."""

import pytest

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.config.runtime import RuntimeConfig
from veeksha.preflight import scorer, validator
from veeksha.preflight.models import MetricSummary, ScoreReport
from veeksha.preflight.report import render_report


def _report(**metric_p99):
    """Build a ScoreReport where each given metric has the specified p99."""
    metrics = {}
    for name, p99 in metric_p99.items():
        metrics[name] = MetricSummary(
            name, 100, p99 / 2, p99, p99 / 2, p99 / 4, 0.0, p99
        )
    return ScoreReport(
        metrics=metrics, n_requests=100, n_paired_requests=100, unpaired_fraction=0.0
    )


_KW = dict(
    delivery_lag_threshold_ms=5.0,
    server_pacing_threshold_ms=5.0,
    dispatch_drift_threshold_ms=10.0,
    input_pacing_threshold_ms=10.0,
    max_unpaired_fraction=0.02,
)


def _all_good():
    return _report(
        **{
            scorer.M_SERVER_TTFC_ABS_ERR: 0.5,
            scorer.M_SERVER_TPOC_ABS_ERR: 0.5,
            scorer.M_REQUEST_DELIVERY: 1.0,
            scorer.M_RESPONSE_DELIVERY: 1.0,
            scorer.M_LIFECYCLE_READY_TO_SEND: 2.0,
        }
    )


def test_all_within_threshold_passes():
    result = validator.run_validation(_all_good(), **_KW)
    assert result.verdict == validator.VERDICT_PASS
    assert result.is_pass
    assert all(g.passed for g in result.gates)


def test_harness_delivery_breach_fails():
    rep = _all_good()
    rep.metrics[scorer.M_RESPONSE_DELIVERY] = MetricSummary(
        scorer.M_RESPONSE_DELIVERY,
        100,
        3.0,
        42.0,
        5.0,
        10.0,
        0.0,
        42.0,  # p99 42ms >> 5
    )
    result = validator.run_validation(rep, **_KW)
    assert result.verdict == validator.VERDICT_FAIL
    assert result.failed_gates("harness")


def test_server_pacing_breach_is_server_at_capacity():
    # Both a server-pacing gate AND a harness gate fail -> SERVER_AT_CAPACITY wins.
    rep = _all_good()
    rep.metrics[scorer.M_SERVER_TPOC_ABS_ERR] = MetricSummary(
        scorer.M_SERVER_TPOC_ABS_ERR, 100, 4.0, 30.0, 6.0, 7.0, 0.0, 30.0
    )
    rep.metrics[scorer.M_REQUEST_DELIVERY] = MetricSummary(
        scorer.M_REQUEST_DELIVERY, 100, 3.0, 40.0, 5.0, 9.0, 0.0, 40.0
    )
    result = validator.run_validation(rep, **_KW)
    assert result.verdict == validator.VERDICT_SERVER_AT_CAPACITY


def test_too_many_unpaired_fails():
    rep = _all_good()
    rep.unpaired_fraction = 0.10  # 10% > 2%
    result = validator.run_validation(rep, **_KW)
    assert result.verdict == validator.VERDICT_FAIL


def test_missing_metric_fails_safe():
    # No metrics at all: NaN p99 fails every harness gate -> FAIL.
    result = validator.run_validation(_report(), **_KW)
    assert result.verdict == validator.VERDICT_FAIL


def test_report_renders_verdict_and_gates():
    result = validator.run_validation(_all_good(), **_KW)
    config = PreflightCheckConfig(
        concurrency=50, runtime=RuntimeConfig(max_sessions=500)
    )
    text = render_report(_all_good(), result, config, config.text)
    assert "VERDICT: PASS" in text
    assert "request delivery" in text
    assert "PASS" in text
    # The run's settings are always recorded.
    assert "Configuration:" in text
    assert "concurrency" in text
    assert "runtime.max_sessions" in text
    assert "server_ttfc_ms" in text
    assert "server_tpoc_ms" in text
