from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass
class InverseNotebookArtifacts:
    output_dir: Path
    episode_analysis: pd.DataFrame
    state_profile: pd.DataFrame
    quality_diagnostics: dict[str, object]
    observation_audit: dict[str, object]
    metadata_extraction_summary: dict[str, object]
    sequence_audit: dict[str, object]
    model_health_summary: dict[str, object]
    observation_mapping_crosstab: pd.DataFrame
    raw_finish_signal_summary: pd.DataFrame
    unsupported_score_values: pd.DataFrame
    sequence_length_distribution: pd.DataFrame
    suspicious_sequences: pd.DataFrame
    report_markdown: str
    plot_paths: dict[str, Path]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_inverse_artifacts(output_dir: str | Path) -> InverseNotebookArtifacts:
    out = Path(output_dir)
    episode_analysis_path = out / "diagnostics" / "episode_analysis.csv"
    state_profile_path = out / "diagnostics" / "state_profile.csv"
    quality_diag_path = out / "diagnostics" / "quality_diagnostics.json"
    observation_audit_path = out / "diagnostics" / "observation_audit.json"
    metadata_summary_path = out / "diagnostics" / "metadata_extraction_summary.json"
    sequence_audit_path = out / "diagnostics" / "sequence_audit.json"
    model_health_path = out / "diagnostics" / "model_health_summary.json"
    observation_crosstab_path = out / "diagnostics" / "observation_mapping_crosstab.csv"
    raw_finish_summary_path = out / "diagnostics" / "raw_finish_signal_summary.csv"
    unsupported_score_path = out / "diagnostics" / "unsupported_score_values.csv"
    sequence_length_path = out / "diagnostics" / "sequence_length_distribution.csv"
    suspicious_sequences_path = out / "diagnostics" / "suspicious_sequences.csv"
    report_path = out / "reports" / "inverse_diagnostic_report.md"

    plot_paths = {
        "hidden_state_sequence": out / "plots" / "hidden_state_sequence.png",
        "state_probability_profile": out / "plots" / "state_probability_profile.png",
        "transition_distribution": out / "plots" / "transition_distribution.png",
    }

    return InverseNotebookArtifacts(
        output_dir=out,
        episode_analysis=_safe_read_csv(episode_analysis_path),
        state_profile=_safe_read_csv(state_profile_path),
        quality_diagnostics=_safe_read_json(quality_diag_path),
        observation_audit=_safe_read_json(observation_audit_path),
        metadata_extraction_summary=_safe_read_json(metadata_summary_path),
        sequence_audit=_safe_read_json(sequence_audit_path),
        model_health_summary=_safe_read_json(model_health_path),
        observation_mapping_crosstab=_safe_read_csv(observation_crosstab_path),
        raw_finish_signal_summary=_safe_read_csv(raw_finish_summary_path),
        unsupported_score_values=_safe_read_csv(unsupported_score_path),
        sequence_length_distribution=_safe_read_csv(sequence_length_path),
        suspicious_sequences=_safe_read_csv(suspicious_sequences_path),
        report_markdown=_safe_read_text(report_path),
        plot_paths={k: v for k, v in plot_paths.items() if v.exists()},
    )


def display_inverse_report(report_markdown: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(report_markdown))
    except Exception:
        print(report_markdown)


def display_inverse_plots(plot_paths: dict[str, Path]) -> None:
    try:
        from IPython.display import Image, Markdown, display

        if not plot_paths:
            display(Markdown("_Plots are not available._"))
            return

        for title, path in plot_paths.items():
            display(Markdown(f"### {title}"))
            display(Image(filename=str(path)))
    except Exception:
        for title, path in plot_paths.items():
            print(f"{title}: {path}")
