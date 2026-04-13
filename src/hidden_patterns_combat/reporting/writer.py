from __future__ import annotations

from pathlib import Path

from .schemas import AnalysisReport


def write_analysis_markdown(report: AnalysisReport, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HMM Analysis Report",
        "",
        f"- Rows: {report.rows}",
        f"- Log-likelihood: {report.log_likelihood:.4f}",
        f"- Episode analysis: `{report.analysis_csv}`",
        f"- State profile: `{report.profile_csv}`",
        f"- Interpretation text: `{report.summary_path}`",
    ]

    if report.plots:
        lines.append("- Plots:")
        for plot_path in report.plots:
            lines.append(f"  - `{plot_path}`")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
