from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class MetadataAuditResult:
    summary: dict[str, Any]


def _missing_share_text(series: pd.Series) -> float:
    text = series.fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})
    if len(text) == 0:
        return 1.0
    return float(text.eq("").mean())


def _zero_share_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(numeric) == 0:
        return 1.0
    return float(numeric.eq(0.0).mean())


def _field_quality(canonical_df: pd.DataFrame, field: str) -> dict[str, Any]:
    if field not in canonical_df.columns:
        return {
            "available_in_canonical": False,
            "missing_share": 1.0,
            "zero_share": None,
            "unique_values": 0,
            "informative": False,
        }

    series = canonical_df[field]
    if field in {"episode_time_sec", "pause_time_sec", "score"}:
        zero_share = _zero_share_numeric(series)
        missing_share = float(pd.to_numeric(series, errors="coerce").isna().mean())
        informative = bool((1.0 - zero_share) >= 0.10)
        unique_values = int(pd.to_numeric(series, errors="coerce").nunique(dropna=True))
        return {
            "available_in_canonical": True,
            "missing_share": missing_share,
            "zero_share": zero_share,
            "unique_values": unique_values,
            "informative": informative,
        }

    missing_share = _missing_share_text(series)
    unique_values = int(
        series.fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""}).nunique(dropna=True)
    )
    informative = bool((1.0 - missing_share) >= 0.10 and unique_values >= 2)
    return {
        "available_in_canonical": True,
        "missing_share": missing_share,
        "zero_share": None,
        "unique_values": unique_values,
        "informative": informative,
    }


def build_metadata_extraction_summary(
    canonical_df: pd.DataFrame,
    extraction_info: dict[str, Any] | None = None,
) -> MetadataAuditResult:
    info = extraction_info or {}
    detected_columns = info.get("selected_columns", {}) or {}
    selection_method = info.get("selection_method", {}) or {}

    target_fields = [
        "athlete_name",
        "athlete_id",
        "sheet_name",
        "weight_class",
        "episode_id",
        "episode_time_sec",
        "pause_time_sec",
        "score",
        "opponent_name",
        "tournament_name",
        "event_date",
        "sequence_id",
    ]

    field_quality = {field: _field_quality(canonical_df, field) for field in target_fields}

    detected_metadata_fields = [
        field
        for field in target_fields
        if field_quality[field]["available_in_canonical"] and field_quality[field]["missing_share"] < 1.0
    ]
    missing_metadata_fields = [
        field
        for field in target_fields
        if (not field_quality[field]["available_in_canonical"]) or field_quality[field]["missing_share"] >= 1.0
    ]

    episode_time_informative = bool(field_quality["episode_time_sec"]["informative"])
    pause_time_informative = bool(field_quality["pause_time_sec"]["informative"])
    weight_class_informative = bool(field_quality["weight_class"]["informative"])

    warnings: list[str] = []
    if "weight_class" in missing_metadata_fields or not weight_class_informative:
        warnings.append("Weight class is missing or non-informative in canonical episodes.")
    if not episode_time_informative:
        warnings.append("Episode time is missing or mostly zero; timing signal is weak.")
    if not pause_time_informative:
        warnings.append("Pause time is missing or mostly zero; pause-based interpretation is limited.")
    for field in ("opponent_name", "tournament_name", "event_date"):
        if field in missing_metadata_fields:
            warnings.append(f"{field} is unavailable; sequence segmentation cannot use this marker.")

    summary: dict[str, Any] = {
        "rows_total": int(len(canonical_df)),
        "detected_source_columns": {str(k): (None if v is None else str(v)) for k, v in detected_columns.items()},
        "selection_method": {str(k): str(v) for k, v in selection_method.items()},
        "field_quality": field_quality,
        "detected_metadata_fields": detected_metadata_fields,
        "missing_metadata_fields": missing_metadata_fields,
        "episode_time_informative": episode_time_informative,
        "pause_time_informative": pause_time_informative,
        "weight_class_informative": weight_class_informative,
        "warnings": warnings,
    }
    return MetadataAuditResult(summary=summary)


def write_metadata_audit(
    result: MetadataAuditResult,
    diagnostics_dir: str | Path,
) -> str:
    out = Path(diagnostics_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / "metadata_extraction_summary.json"
    path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)
