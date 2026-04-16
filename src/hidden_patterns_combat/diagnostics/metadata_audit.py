from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


_NUMERIC_FIELDS = {"episode_time_sec", "pause_time_sec", "score"}


@dataclass
class MetadataAuditResult:
    summary: dict[str, Any]
    field_coverage: pd.DataFrame


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})


def _missing_share_text(series: pd.Series) -> float:
    text = _normalized_text(series)
    if len(text) == 0:
        return 1.0
    return float(text.eq("").mean())


def _zero_share_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(numeric) == 0:
        return 1.0
    return float(numeric.eq(0.0).mean())


def _sample_values(series: pd.Series, *, max_values: int = 5) -> list[str]:
    text = _normalized_text(series)
    text = text[text != ""]
    if text.empty:
        return []
    values = text.drop_duplicates().head(max_values).tolist()
    return [str(v) for v in values]


def _field_quality(
    canonical_df: pd.DataFrame,
    field: str,
    *,
    source_column: str | None,
    selection_method: str | None,
) -> dict[str, Any]:
    if field not in canonical_df.columns:
        return {
            "available_in_canonical": False,
            "source_column": None if source_column is None else str(source_column),
            "selection_method": None if selection_method is None else str(selection_method),
            "missing_share": 1.0,
            "zero_share": None,
            "unique_values": 0,
            "informative": False,
            "sample_values": [],
        }

    series = canonical_df[field]
    if field in _NUMERIC_FIELDS:
        numeric = pd.to_numeric(series, errors="coerce")
        missing_share = float(numeric.isna().mean())
        zero_share = _zero_share_numeric(series)
        unique_values = int(numeric.nunique(dropna=True))
        informative = bool((1.0 - zero_share) >= 0.10 and unique_values >= 2)
        return {
            "available_in_canonical": True,
            "source_column": None if source_column is None else str(source_column),
            "selection_method": None if selection_method is None else str(selection_method),
            "missing_share": missing_share,
            "zero_share": zero_share,
            "unique_values": unique_values,
            "informative": informative,
            "sample_values": _sample_values(series),
        }

    missing_share = _missing_share_text(series)
    unique_values = int(_normalized_text(series)[lambda s: s != ""].nunique(dropna=True))
    informative = bool((1.0 - missing_share) >= 0.10 and unique_values >= 2)
    return {
        "available_in_canonical": True,
        "source_column": None if source_column is None else str(source_column),
        "selection_method": None if selection_method is None else str(selection_method),
        "missing_share": missing_share,
        "zero_share": None,
        "unique_values": unique_values,
        "informative": informative,
        "sample_values": _sample_values(series),
    }


def _support_level(strong_condition: bool, partial_condition: bool) -> str:
    if strong_condition:
        return "strong"
    if partial_condition:
        return "partial"
    return "weak"


def build_metadata_extraction_summary(
    canonical_df: pd.DataFrame,
    extraction_info: dict[str, Any] | None = None,
) -> MetadataAuditResult:
    info = extraction_info or {}
    detected_columns = info.get("selected_columns", {}) or {}
    selection_method = info.get("selection_method", {}) or {}
    field_candidates = info.get("field_candidates", {}) or {}

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

    field_quality: dict[str, dict[str, Any]] = {}
    for field in target_fields:
        field_quality[field] = _field_quality(
            canonical_df,
            field,
            source_column=detected_columns.get(field),
            selection_method=selection_method.get(field),
        )

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

    informative_fields = [field for field in target_fields if bool(field_quality[field]["informative"])]
    non_informative_fields = [field for field in target_fields if not bool(field_quality[field]["informative"])]

    episode_time_informative = bool(field_quality["episode_time_sec"]["informative"])
    pause_time_informative = bool(field_quality["pause_time_sec"]["informative"])
    weight_class_informative = bool(field_quality["weight_class"]["informative"])

    sequence_id_informative = bool(field_quality["sequence_id"]["informative"])
    segmentation_context_fields = ["weight_class", "opponent_name", "tournament_name", "event_date"]
    segmentation_context_informative = [
        field for field in segmentation_context_fields if bool(field_quality[field]["informative"])
    ]
    segmentation_context_available = [
        field
        for field in segmentation_context_fields
        if bool(field_quality[field]["available_in_canonical"]) and float(field_quality[field]["missing_share"]) < 1.0
    ]

    explicit_sequence_source = detected_columns.get("explicit_sequence_id")
    explicit_sequence_available = bool(str(explicit_sequence_source).strip()) if explicit_sequence_source else False

    segmentation_readiness = _support_level(
        strong_condition=sequence_id_informative and len(segmentation_context_informative) >= 2,
        partial_condition=sequence_id_informative,
    )

    temporal_support_level = _support_level(
        strong_condition=episode_time_informative and pause_time_informative,
        partial_condition=episode_time_informative or pause_time_informative,
    )

    semantic_support_level = _support_level(
        strong_condition=weight_class_informative and sequence_id_informative,
        partial_condition=weight_class_informative or sequence_id_informative,
    )

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
    if segmentation_readiness == "weak":
        warnings.append("Metadata support for sequence segmentation is weak.")

    critical_fields = [
        "weight_class",
        "episode_time_sec",
        "pause_time_sec",
        "opponent_name",
        "tournament_name",
        "event_date",
        "sequence_id",
    ]
    critical_field_quality = {
        field: {
            "found": bool(field_quality[field]["available_in_canonical"])
            and float(field_quality[field]["missing_share"]) < 1.0,
            "source_column": field_quality[field]["source_column"],
            "selection_method": field_quality[field]["selection_method"],
            "missing_share": float(field_quality[field]["missing_share"]),
            "zero_share": field_quality[field]["zero_share"],
            "informative": bool(field_quality[field]["informative"]),
            "sample_values": list(field_quality[field]["sample_values"]),
        }
        for field in critical_fields
    }

    coverage_rows: list[dict[str, Any]] = []
    for field in target_fields:
        quality = field_quality[field]
        coverage_rows.append(
            {
                "field": field,
                "source_column": quality.get("source_column"),
                "selection_method": quality.get("selection_method"),
                "available_in_canonical": bool(quality.get("available_in_canonical", False)),
                "missing_share": float(quality.get("missing_share", 1.0)),
                "zero_share": quality.get("zero_share"),
                "unique_values": int(quality.get("unique_values", 0)),
                "informative": bool(quality.get("informative", False)),
                "sample_values": json.dumps(quality.get("sample_values", []), ensure_ascii=False),
            }
        )
    field_coverage = pd.DataFrame(coverage_rows).sort_values("field").reset_index(drop=True)

    summary: dict[str, Any] = {
        "rows_total": int(len(canonical_df)),
        "detected_source_columns": {str(k): (None if v is None else str(v)) for k, v in detected_columns.items()},
        "selection_method": {str(k): str(v) for k, v in selection_method.items()},
        "field_candidates": {str(k): [str(x) for x in (v or [])] for k, v in field_candidates.items()},
        "field_quality": field_quality,
        "detected_metadata_fields": detected_metadata_fields,
        "missing_metadata_fields": missing_metadata_fields,
        "informative_fields": informative_fields,
        "non_informative_fields": non_informative_fields,
        "episode_time_informative": episode_time_informative,
        "pause_time_informative": pause_time_informative,
        "weight_class_informative": weight_class_informative,
        "segmentation_support": {
            "readiness": segmentation_readiness,
            "explicit_sequence_field_available": explicit_sequence_available,
            "explicit_sequence_source_column": None if explicit_sequence_source is None else str(explicit_sequence_source),
            "sequence_id_informative": sequence_id_informative,
            "context_fields_available": segmentation_context_available,
            "context_fields_missing": [
                field
                for field in segmentation_context_fields
                if field not in segmentation_context_available
            ],
            "context_fields_informative": segmentation_context_informative,
            "context_fields_non_informative": [
                field for field in segmentation_context_fields if field not in segmentation_context_informative
            ],
        },
        "temporal_modeling_support": {
            "readiness": temporal_support_level,
            "episode_time_informative": episode_time_informative,
            "pause_time_informative": pause_time_informative,
        },
        "semantic_interpretation_support": {
            "readiness": semantic_support_level,
            "weight_class_informative": weight_class_informative,
            "sequence_id_informative": sequence_id_informative,
        },
        "critical_field_quality": critical_field_quality,
        "warnings": warnings,
    }
    return MetadataAuditResult(summary=summary, field_coverage=field_coverage)


def write_metadata_audit(
    result: MetadataAuditResult,
    diagnostics_dir: str | Path,
) -> str:
    out = Path(diagnostics_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / "metadata_extraction_summary.json"
    coverage_path = out / "metadata_field_coverage.csv"
    path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.field_coverage.to_csv(coverage_path, index=False)
    return str(path)
