from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass
class InverseNotebookArtifacts:
    output_dir: Path
    run_summary: dict[str, object]
    episode_analysis: pd.DataFrame
    state_profile: pd.DataFrame
    quality_diagnostics: dict[str, object]
    observation_audit: dict[str, object]
    metadata_extraction_summary: dict[str, object]
    sequence_audit: dict[str, object]
    model_health_summary: dict[str, object]
    observation_mapping_crosstab: pd.DataFrame
    raw_finish_signal_summary: pd.DataFrame
    unsupported_finish_values: pd.DataFrame
    unsupported_score_values: pd.DataFrame
    metadata_field_coverage: pd.DataFrame
    sequence_length_distribution: pd.DataFrame
    suspicious_sequences: pd.DataFrame
    report_markdown: str
    plot_paths: dict[str, Path]
    artifact_status: pd.DataFrame
    loader_warnings: list[str]


def _read_csv_with_status(path: Path) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        return pd.DataFrame(), "missing", "file not found"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), "invalid", f"csv read error: {exc}"
    if frame.empty:
        return frame, "empty", "csv has header but no rows"
    return frame, "ok", f"rows={len(frame)}"


def _read_json_with_status(path: Path) -> tuple[dict[str, object], str, str]:
    if not path.exists():
        return {}, "missing", "file not found"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, "invalid", f"json parse error: {exc}"
    if not isinstance(payload, dict):
        return {}, "invalid", "json payload is not an object"
    if not payload:
        return payload, "empty", "json object is empty"
    return payload, "ok", f"keys={len(payload)}"


def _read_text_with_status(path: Path) -> tuple[str, str, str]:
    if not path.exists():
        return "", "missing", "file not found"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return "", "invalid", f"text read error: {exc}"
    if not text.strip():
        return text, "empty", "text file is empty"
    return text, "ok", f"chars={len(text)}"


def _status_row(artifact_name: str, path: Path, artifact_type: str, status: str, detail: str) -> dict[str, object]:
    return {
        "artifact_name": artifact_name,
        "artifact_type": artifact_type,
        "path": str(path),
        "status": status,
        "detail": detail,
    }


def load_inverse_artifacts(output_dir: str | Path) -> InverseNotebookArtifacts:
    out = Path(output_dir)
    status_rows: list[dict[str, object]] = []
    loader_warnings: list[str] = []

    def _record(artifact_name: str, path: Path, artifact_type: str, status: str, detail: str) -> None:
        status_rows.append(_status_row(artifact_name, path, artifact_type, status, detail))
        if status != "ok":
            loader_warnings.append(f"{artifact_name}: {status} ({detail}) -> {path}")

    def _load_csv(artifact_name: str, path: Path) -> pd.DataFrame:
        value, status, detail = _read_csv_with_status(path)
        _record(artifact_name, path, "csv", status, detail)
        return value

    def _load_json(artifact_name: str, path: Path) -> dict[str, object]:
        value, status, detail = _read_json_with_status(path)
        _record(artifact_name, path, "json", status, detail)
        return value

    def _load_text(artifact_name: str, path: Path) -> str:
        value, status, detail = _read_text_with_status(path)
        _record(artifact_name, path, "text", status, detail)
        return value

    episode_analysis_path = out / "diagnostics" / "episode_analysis.csv"
    state_profile_path = out / "diagnostics" / "state_profile.csv"
    run_summary_path = out / "diagnostics" / "run_summary.json"
    quality_diag_path = out / "diagnostics" / "quality_diagnostics.json"
    observation_audit_path = out / "diagnostics" / "observation_audit.json"
    metadata_summary_path = out / "diagnostics" / "metadata_extraction_summary.json"
    sequence_audit_path = out / "diagnostics" / "sequence_audit.json"
    model_health_path = out / "diagnostics" / "model_health_summary.json"
    observation_crosstab_path = out / "diagnostics" / "observation_mapping_crosstab.csv"
    raw_finish_summary_path = out / "diagnostics" / "raw_finish_signal_summary.csv"
    unsupported_finish_path = out / "diagnostics" / "unsupported_finish_values.csv"
    unsupported_score_path = out / "diagnostics" / "unsupported_score_values.csv"
    metadata_field_coverage_path = out / "diagnostics" / "metadata_field_coverage.csv"
    sequence_length_path = out / "diagnostics" / "sequence_length_distribution.csv"
    suspicious_sequences_path = out / "diagnostics" / "suspicious_sequences.csv"
    report_path = out / "reports" / "inverse_diagnostic_report.md"

    run_summary = _load_json("run_summary", run_summary_path)
    episode_analysis = _load_csv("episode_analysis", episode_analysis_path)
    state_profile = _load_csv("state_profile", state_profile_path)
    quality_diagnostics = _load_json("quality_diagnostics", quality_diag_path)
    if not run_summary:
        fallback_run_summary = quality_diagnostics.get("run_summary", {})
        if isinstance(fallback_run_summary, dict) and fallback_run_summary:
            run_summary = {str(k): v for k, v in fallback_run_summary.items()}
    observation_audit = _load_json("observation_audit", observation_audit_path)
    metadata_extraction_summary = _load_json("metadata_extraction_summary", metadata_summary_path)
    sequence_audit = _load_json("sequence_audit", sequence_audit_path)
    model_health_summary = _load_json("model_health_summary", model_health_path)
    observation_mapping_crosstab = _load_csv("observation_mapping_crosstab", observation_crosstab_path)
    raw_finish_signal_summary = _load_csv("raw_finish_signal_summary", raw_finish_summary_path)
    unsupported_finish_values = _load_csv("unsupported_finish_values", unsupported_finish_path)
    unsupported_score_values = _load_csv("unsupported_score_values", unsupported_score_path)
    metadata_field_coverage = _load_csv("metadata_field_coverage", metadata_field_coverage_path)
    sequence_length_distribution = _load_csv("sequence_length_distribution", sequence_length_path)
    suspicious_sequences = _load_csv("suspicious_sequences", suspicious_sequences_path)
    report_markdown = _load_text("report_markdown", report_path)

    plot_paths = {
        "hidden_state_sequence": out / "plots" / "hidden_state_sequence.png",
        "state_probability_profile": out / "plots" / "state_probability_profile.png",
        "transition_distribution": out / "plots" / "transition_distribution.png",
    }

    artifact_status = pd.DataFrame(status_rows)
    if not artifact_status.empty:
        artifact_status = artifact_status.sort_values(["status", "artifact_name"]).reset_index(drop=True)

    return InverseNotebookArtifacts(
        output_dir=out,
        run_summary=run_summary,
        episode_analysis=episode_analysis,
        state_profile=state_profile,
        quality_diagnostics=quality_diagnostics,
        observation_audit=observation_audit,
        metadata_extraction_summary=metadata_extraction_summary,
        sequence_audit=sequence_audit,
        model_health_summary=model_health_summary,
        observation_mapping_crosstab=observation_mapping_crosstab,
        raw_finish_signal_summary=raw_finish_signal_summary,
        unsupported_finish_values=unsupported_finish_values,
        unsupported_score_values=unsupported_score_values,
        metadata_field_coverage=metadata_field_coverage,
        sequence_length_distribution=sequence_length_distribution,
        suspicious_sequences=suspicious_sequences,
        report_markdown=report_markdown,
        plot_paths={k: v for k, v in plot_paths.items() if v.exists()},
        artifact_status=artifact_status,
        loader_warnings=loader_warnings,
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
