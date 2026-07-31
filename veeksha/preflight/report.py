"""Render a preflight run (scores + verdict) as a plain-text report."""

from __future__ import annotations

import math
import os
from typing import List

from veeksha.config.preflight import PreflightCheckConfig
from veeksha.preflight.models import ScoreReport
from veeksha.preflight.validator import ValidationResult

_REPORT_FILENAME = "preflight_report.txt"


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def _table(rows: List[List[str]], headers: List[str]) -> str:
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for row in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def _config_rows(config: PreflightCheckConfig, check_config) -> List[List[str]]:
    """Flatten the load level and workload/mock timings into name/value rows."""
    rows = [
        ["concurrency", str(config.concurrency)],
    ]
    for name, value in vars(config.runtime).items():
        if not name.startswith("_"):
            rows.append([f"runtime.{name}", str(value)])
    for name, value in vars(check_config).items():
        if not name.startswith("_"):
            rows.append([name, str(value)])
    return rows


def render_report(
    score_report: ScoreReport,
    validation: ValidationResult,
    config: PreflightCheckConfig,
    check_config,
    *,
    title: str = "Preflight measurement-fidelity check",
) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"VERDICT: {validation.verdict.upper()}")
    lines.append(
        f"requests: {score_report.n_requests}  "
        f"paired: {score_report.n_paired_requests}  "
        f"unpaired: {score_report.unpaired_fraction:.3%}"
    )
    lines.append("")
    lines.append("Configuration:")
    lines.append(_table(_config_rows(config, check_config), ["setting", "value"]))
    lines.append("")

    # --- gates ---
    lines.append("Gates (p99 vs threshold):")
    gate_rows = [
        [
            g.name,
            g.category,
            _fmt(g.p99),
            _fmt(g.threshold),
            "PASS" if g.passed else "FAIL",
        ]
        for g in validation.gates
    ]
    lines.append(_table(gate_rows, ["gate", "blames", "p99", "threshold", "result"]))
    lines.append("")

    # --- full metric distributions ---
    lines.append("Metrics (ms):")
    metric_rows = []
    for name in sorted(score_report.metrics):
        m = score_report.metrics[name]
        metric_rows.append(
            [
                name,
                str(m.count),
                _fmt(m.p50),
                _fmt(m.p99),
                _fmt(m.stdev),
                _fmt(m.mean),
                _fmt(m.maximum),
            ]
        )
    lines.append(
        _table(metric_rows, ["metric", "count", "p50", "p99", "jitter", "mean", "max"])
    )
    lines.append("")
    return "\n".join(lines)


def write_report(text: str, output_dir: str) -> str:
    """Write the rendered report to ``output_dir`` and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, _REPORT_FILENAME)
    with open(path, "w") as f:
        f.write(text)
    return path
