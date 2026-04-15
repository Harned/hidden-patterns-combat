from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SequenceAuditResult:
    summary: dict[str, Any]
    length_distribution: pd.DataFrame
    suspicious_sequences: pd.DataFrame


def _normalize_sequence_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("sequence_0")
        .astype(str)
        .str.strip()
        .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
    )


def _safe_mode(series: pd.Series, default: str = "unknown") -> str:
    if series.empty:
        return default
    mode = series.mode(dropna=True)
    if mode.empty:
        return default
    return str(mode.iloc[0])


def build_sequence_audit(analysis_df: pd.DataFrame) -> SequenceAuditResult:
    frame = analysis_df.copy().reset_index(drop=True)
    if frame.empty:
        empty_len = pd.DataFrame(columns=["sequence_length", "sequence_count", "share_of_sequences"])
        empty_susp = pd.DataFrame(columns=["sequence_id", "sequence_length", "suspicion_reasons"])
        summary = {
            "rows_total": 0,
            "n_sequences": 0,
            "explicit_sequence_share": 0.0,
            "surrogate_sequence_share": 0.0,
            "fallback_sequence_share": 0.0,
            "high_quality_share": 0.0,
            "medium_quality_share": 0.0,
            "low_quality_share": 0.0,
            "sequence_length_stats": {"min": 0, "max": 0, "median": 0.0, "mean": 0.0, "p95": 0.0},
            "long_sequence_threshold": 0,
            "suspicious_long_sequences": 0,
            "suspicious_flat_sequences": 0,
            "explicit_sequence_available": False,
            "explicit_unavailable_reasons": [],
            "warnings": ["No rows in analysis dataframe for sequence audit."],
        }
        return SequenceAuditResult(summary=summary, length_distribution=empty_len, suspicious_sequences=empty_susp)

    frame["sequence_id"] = _normalize_sequence_id(
        frame.get("sequence_id", pd.Series(["sequence_0"] * len(frame), index=frame.index))
    )
    frame["sequence_resolution_type"] = frame.get(
        "sequence_resolution_type",
        pd.Series(["fallback"] * len(frame), index=frame.index),
    ).astype(str)
    frame["sequence_quality_flag"] = frame.get(
        "sequence_quality_flag",
        pd.Series(["low"] * len(frame), index=frame.index),
    ).astype(str)
    frame["observed_zap_class"] = frame.get(
        "observed_zap_class",
        pd.Series(["unknown"] * len(frame), index=frame.index),
    ).astype(str)

    length_series = frame.groupby("sequence_id", dropna=False).size().astype(int)
    n_sequences = int(len(length_series))
    length_distribution = (
        length_series.value_counts()
        .sort_index()
        .rename_axis("sequence_length")
        .reset_index(name="sequence_count")
    )
    length_distribution["share_of_sequences"] = (
        length_distribution["sequence_count"].astype(float) / max(1, n_sequences)
    )

    grouped_rows: list[dict[str, Any]] = []
    for sequence_id, group in frame.groupby("sequence_id", dropna=False, sort=False):
        observed_unique = int(group["observed_zap_class"].nunique(dropna=True))
        hidden_unique = (
            int(group["hidden_state"].nunique(dropna=True))
            if "hidden_state" in group.columns
            else 0
        )
        grouped_rows.append(
            {
                "sequence_id": str(sequence_id),
                "sequence_length": int(len(group)),
                "resolution_type": _safe_mode(group["sequence_resolution_type"]),
                "quality_flag": _safe_mode(group["sequence_quality_flag"]),
                "observed_unique_classes": observed_unique,
                "hidden_unique_states": hidden_unique,
                "no_score_share": float((group["observed_zap_class"] == "no_score").mean()),
                "unknown_share": float((group["observed_zap_class"] == "unknown").mean()),
            }
        )
    sequence_table = pd.DataFrame(grouped_rows)

    p95 = float(length_series.quantile(0.95)) if not length_series.empty else 0.0
    long_threshold = max(25, int(np.ceil(p95)))

    suspicious_rows: list[dict[str, Any]] = []
    for _, row in sequence_table.iterrows():
        reasons: list[str] = []
        if int(row["sequence_length"]) >= long_threshold:
            reasons.append("suspiciously_long_sequence")
        if int(row["observed_unique_classes"]) <= 1 and int(row["hidden_unique_states"]) <= 1:
            reasons.append("flat_observed_and_hidden")
        if float(row["no_score_share"]) >= 0.95 and int(row["observed_unique_classes"]) <= 1:
            reasons.append("mostly_no_score_flat_sequence")
        if str(row["quality_flag"]) == "low":
            reasons.append("low_quality_segmentation")
        if not reasons:
            continue
        suspicious_rows.append(
            {
                **row.to_dict(),
                "suspicion_reasons": ";".join(sorted(set(reasons))),
            }
        )
    suspicious_sequences = (
        pd.DataFrame(suspicious_rows)
        if suspicious_rows
        else pd.DataFrame(
            columns=[
                "sequence_id",
                "sequence_length",
                "resolution_type",
                "quality_flag",
                "observed_unique_classes",
                "hidden_unique_states",
                "no_score_share",
                "unknown_share",
                "suspicion_reasons",
            ]
        )
    )
    if not suspicious_sequences.empty:
        suspicious_sequences = suspicious_sequences.sort_values(
            ["sequence_length", "sequence_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

    rows_total = max(1, int(len(frame)))
    explicit_share = float((frame["sequence_resolution_type"] == "explicit").mean())
    surrogate_share = float((frame["sequence_resolution_type"] == "surrogate").mean())
    fallback_share = float((frame["sequence_resolution_type"] == "fallback").mean())

    quality_high = float((frame["sequence_quality_flag"] == "high").mean())
    quality_medium = float((frame["sequence_quality_flag"] == "medium").mean())
    quality_low = float((frame["sequence_quality_flag"] == "low").mean())

    explicit_reasons: list[str] = []
    if explicit_share <= 0.0:
        reason_series = frame.get(
            "sequence_quality_reason",
            pd.Series(["explicit_sequence_unavailable"] * len(frame), index=frame.index),
        ).astype(str)
        counts = reason_series.value_counts().head(8)
        explicit_reasons = [f"{idx}: {int(val)}" for idx, val in counts.items()]

    warnings: list[str] = []
    if explicit_share <= 0.0:
        warnings.append(
            "Explicit bout/sequence ids were not detected; segmentation is surrogate-based."
        )
    if surrogate_share >= 0.80:
        warnings.append("Most rows rely on surrogate sequence segmentation.")
    if quality_low >= 0.30:
        warnings.append("A substantial share of rows has low sequence quality.")
    if not suspicious_sequences.empty:
        warnings.append("Suspicious sequences were detected; inspect suspicious_sequences.csv.")

    summary: dict[str, Any] = {
        "rows_total": int(len(frame)),
        "n_sequences": n_sequences,
        "explicit_sequence_share": explicit_share,
        "surrogate_sequence_share": surrogate_share,
        "fallback_sequence_share": fallback_share,
        "high_quality_share": quality_high,
        "medium_quality_share": quality_medium,
        "low_quality_share": quality_low,
        "sequence_length_stats": {
            "min": int(length_series.min()) if not length_series.empty else 0,
            "max": int(length_series.max()) if not length_series.empty else 0,
            "median": float(length_series.median()) if not length_series.empty else 0.0,
            "mean": float(length_series.mean()) if not length_series.empty else 0.0,
            "p95": p95,
        },
        "long_sequence_threshold": int(long_threshold),
        "suspicious_long_sequences": int(
            (suspicious_sequences["suspicion_reasons"].astype(str).str.contains("suspiciously_long_sequence")).sum()
        )
        if not suspicious_sequences.empty
        else 0,
        "suspicious_flat_sequences": int(
            (suspicious_sequences["suspicion_reasons"].astype(str).str.contains("flat_observed_and_hidden")).sum()
        )
        if not suspicious_sequences.empty
        else 0,
        "explicit_sequence_available": bool(explicit_share > 0.0),
        "explicit_unavailable_reasons": explicit_reasons,
        "warnings": warnings,
    }

    return SequenceAuditResult(
        summary=summary,
        length_distribution=length_distribution,
        suspicious_sequences=suspicious_sequences,
    )


def write_sequence_audit(
    result: SequenceAuditResult,
    diagnostics_dir: str | Path,
) -> dict[str, str]:
    out = Path(diagnostics_dir)
    out.mkdir(parents=True, exist_ok=True)

    sequence_audit_path = out / "sequence_audit.json"
    length_distribution_path = out / "sequence_length_distribution.csv"
    suspicious_path = out / "suspicious_sequences.csv"

    sequence_audit_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.length_distribution.to_csv(length_distribution_path, index=False)
    result.suspicious_sequences.to_csv(suspicious_path, index=False)

    return {
        "sequence_audit_json": str(sequence_audit_path),
        "sequence_length_distribution_csv": str(length_distribution_path),
        "suspicious_sequences_csv": str(suspicious_path),
    }
