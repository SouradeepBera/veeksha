"""CLI entry point for ``veeksha preflight``.

Runs each configured check, scores the timing drift, gates it into a verdict,
prints and writes a per-check report (under ``<output_dir>/<check>/``), and
exits non-zero unless every check comes back honest.
"""

from __future__ import annotations

import os
import sys
from typing import List

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.logger import init_logger
from veeksha.preflight import validator
from veeksha.preflight.drivers import (
    run_completions_preflight,
    run_realtime_tts_preflight,
    run_stt_preflight,
    run_text_preflight,
    run_tts_preflight,
    run_vajra_tts_preflight,
)
from veeksha.preflight.models import ScoreReport
from veeksha.preflight.report import render_report, write_report

logger = init_logger(__name__)


def _gate_and_render(
    config: PreflightCheckConfig,
    report: ScoreReport,
    title: str,
    output_dir: str,
    check_config,
) -> bool:
    """Validate one check's report, print + write it, return True iff PASS."""
    result = validator.run_validation(
        report,
        delivery_lag_threshold_ms=config.delivery_lag_threshold_ms,
        server_pacing_threshold_ms=config.server_pacing_threshold_ms,
        dispatch_drift_threshold_ms=config.dispatch_drift_threshold_ms,
        input_pacing_threshold_ms=config.input_pacing_threshold_ms,
        max_unpaired_fraction=config.max_unpaired_fraction,
    )
    text = render_report(report, result, config, check_config, title=title)
    path = write_report(text, output_dir)
    print(text)
    print(f"Report written to {path}")
    return result.is_pass


def _run_one_check(config: PreflightCheckConfig) -> bool:
    """Run every enabled check for one config; return True iff all passed.

    Each check gets its own ``<output_dir>/<slug>/`` subdirectory (report +
    metrics) so the per-modality outputs don't overwrite one another.
    """
    all_passed = True
    common = dict(
        traffic_scheduler=config.build_traffic(),
        runtime=config.runtime,
    )

    # (enabled, runner, group config, dir slug, display name)
    checks = [
        (config.check_text, run_text_preflight, config.text, "chat", "text (chat)"),
        (
            config.check_completions,
            run_completions_preflight,
            config.text,
            "completions",
            "completions",
        ),
        (config.check_tts, run_tts_preflight, config.tts, "tts", "tts"),
        (
            config.check_realtime_tts,
            run_realtime_tts_preflight,
            config.tts,
            "realtime_tts",
            "realtime_tts",
        ),
        (
            config.check_vajra_tts,
            run_vajra_tts_preflight,
            config.tts,
            "vajra_tts_stream",
            "vajra_tts_stream",
        ),
        (config.check_stt, run_stt_preflight, config.stt, "stt", "stt"),
    ]
    for enabled, runner, group_cfg, slug, name in checks:
        if not enabled:
            continue
        logger.info("Running preflight %s check", name)
        check_output_dir = os.path.join(config.output_dir, slug)
        report = runner(group_cfg, output_dir=check_output_dir, **common)
        all_passed &= _gate_and_render(
            config,
            report,
            f"Preflight measurement fidelity: {name}",
            check_output_dir,
            check_config=group_cfg,
        )

    return all_passed


def run_preflight_cli(configs: List[PreflightCheckConfig]) -> None:
    overall_passed = True
    for config in configs:
        overall_passed = _run_one_check(config) and overall_passed

    if not overall_passed:
        # Non-zero exit so CI/scripts can gate a benchmark on a clean preflight.
        # FAIL and SERVER_AT_CAPACITY both count as "did not pass".
        sys.exit(1)
