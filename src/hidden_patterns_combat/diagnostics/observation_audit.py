from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from hidden_patterns_combat.preprocessing.observation_builder import ObservationMappingConfig


_MISSING_TEXT = {"", "nan", "none", "<na>"}


@dataclass
class ObservationAuditResult:
    summary: dict[str, Any]
    mapping_crosstab: pd.DataFrame
    raw_finish_signal_summary: pd.DataFrame
    unsupported_finish_values: pd.DataFrame
    unsupported_score_values: pd.DataFrame


def _normalize_col(value: object) -> str:
    return str(value).strip().lower().replace("ё", "е")


def _normalize_value(value: object) -> str:
    return str(value).strip()


def _is_non_empty(value: object) -> bool:
    text = _normalize_value(value).lower()
    return text not in _MISSING_TEXT


def _is_positive(value: object) -> bool:
    text = _normalize_value(value).lower()
    if text in _MISSING_TEXT:
        return False
    if text in {"да", "yes", "true", "истина", "y"}:
        return True
    if text in {"нет", "no", "false", "ложь", "n"}:
        return False
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return float(numeric) > 0.0
    return False


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        low = str(candidate).lower()
        if low in lowered:
            return str(lowered[low])
        for col in df.columns:
            if low in str(col).lower():
                return str(col)
    return None


def _finish_position_from_column(col: str) -> int | None:
    normalized = _normalize_col(col)
    for pattern in (
        r"finish_action_(\d+)",
        r"outcome_action_indicator_(\d+)",
        r"finish_action(\d+)",
        r"outcome_action(\d+)",
    ):
        match = re.search(pattern, normalized)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _column_mapped_classes(col: str, cfg: ObservationMappingConfig) -> list[str]:
    normalized = _normalize_col(col)
    mapped: set[str] = set()
    for rule in cfg.finish_rules:
        all_ok = all(token in normalized for token in rule.match_all_tokens) if rule.match_all_tokens else False
        any_ok = any(token in normalized for token in rule.match_any_tokens) if rule.match_any_tokens else False
        token_match = False
        if rule.match_all_tokens and rule.match_any_tokens:
            token_match = all_ok or any_ok
        elif rule.match_all_tokens:
            token_match = all_ok
        elif rule.match_any_tokens:
            token_match = any_ok
        if token_match:
            mapped.add(rule.class_name)

    pos = _finish_position_from_column(col)
    if pos is not None:
        mapped_from_pos = cfg.finish_position_to_class.get(pos)
        if mapped_from_pos:
            mapped.add(str(mapped_from_pos))
    return sorted(mapped)


def _finish_like_columns(df: pd.DataFrame) -> list[str]:
    keywords = (
        "finish",
        "outcome_action",
        "action_indicator",
        "заверша",
        "удерж",
        "болев",
        "submission",
        "throw",
        "брос",
    )
    cols = [
        str(col)
        for col in df.columns
        if any(token in _normalize_col(col) for token in keywords)
    ]
    return sorted(set(cols))


def _column_non_empty_share(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.map(_is_non_empty).mean())


def _column_positive_share(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.map(_is_positive).mean())


def _non_empty_value_counts(series: pd.Series, *, top_k: int = 12) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    text = text[text != ""]
    if text.empty:
        return pd.Series(dtype="int64")
    return text.value_counts().head(top_k)


def _top_values(series: pd.Series, top_k: int = 8) -> list[str]:
    counts = _non_empty_value_counts(series, top_k=top_k)
    return [str(idx) for idx in counts.index.tolist()]


def _value_frequencies_json(series: pd.Series, *, top_k: int = 12) -> str:
    counts = _non_empty_value_counts(series, top_k=top_k)
    payload = [{"value": str(idx), "rows": int(val)} for idx, val in counts.items()]
    return json.dumps(payload, ensure_ascii=False)


def _json_list(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x) for x in parsed]


def _make_mapping_crosstab(observation_df: pd.DataFrame) -> pd.DataFrame:
    rows = observation_df.copy().reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "bridge_source",
                "observation_resolution_type",
                "observed_zap_class",
                "rows",
                "share",
            ]
        )

    def _bridge_source(row: pd.Series) -> str:
        resolution = str(row.get("observation_resolution_type", "unknown"))
        if resolution == "direct_finish_signal":
            classes = _json_list(row.get("finish_match_classes", "[]"))
            if classes:
                return "finish:" + "+".join(sorted(set(classes)))
            cols = _json_list(row.get("finish_match_columns", "[]"))
            if cols:
                return "finish_col:" + "|".join(sorted(set(cols))[:3])
            return "finish:detected"
        if resolution == "inferred_from_score":
            rounded = row.get("score_rounded")
            if pd.notna(rounded):
                return f"score:{int(float(rounded))}"
            return "score:unknown"
        if resolution == "no_score_rule":
            return "score:0"
        quality = str(row.get("observation_quality_flag", "unknown"))
        return f"{resolution}:{quality}"

    rows["bridge_source"] = rows.apply(_bridge_source, axis=1)

    grouped = (
        rows.groupby(
            ["bridge_source", "observation_resolution_type", "observed_zap_class"],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    total = max(1, int(grouped["rows"].sum()))
    grouped["share"] = grouped["rows"].astype(float) / float(total)
    return grouped.reset_index(drop=True)


def _direct_finish_match_counts(observation_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    frame = observation_df.copy().reset_index(drop=True)
    if frame.empty or "observation_resolution_type" not in frame.columns:
        return {}, {}

    direct = frame[frame["observation_resolution_type"].astype(str) == "direct_finish_signal"].copy()
    if direct.empty:
        return {}, {}

    class_counts = (
        direct.get("observed_zap_class", pd.Series(dtype="object"))
        .astype(str)
        .value_counts()
        .to_dict()
    )

    column_counts: dict[str, int] = {}
    for value in direct.get("finish_match_columns", pd.Series(["[]"] * len(direct))).astype(str):
        for col in _json_list(value):
            column_counts[col] = column_counts.get(col, 0) + 1

    class_counts_out = {str(k): int(v) for k, v in class_counts.items()}
    column_counts_out = {str(k): int(v) for k, v in sorted(column_counts.items(), key=lambda item: (-item[1], item[0]))}
    return class_counts_out, column_counts_out


def build_observation_audit(
    cleaned_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    cfg: ObservationMappingConfig,
    *,
    score_column: str | None = None,
    finish_signal_columns: tuple[str, ...] | None = None,
) -> ObservationAuditResult:
    frame = cleaned_df.copy().reset_index(drop=True)
    obs = observation_df.copy().reset_index(drop=True)

    score_col = score_column or _first_existing_column(frame, cfg.score_column_candidates)

    supported_finish_columns = set(finish_signal_columns or ())
    if not supported_finish_columns:
        for col in frame.columns:
            if _column_mapped_classes(str(col), cfg):
                supported_finish_columns.add(str(col))

    finish_candidates = sorted(set(_finish_like_columns(frame)) | supported_finish_columns)

    finish_rows: list[dict[str, Any]] = []
    for col in finish_candidates:
        if col not in frame.columns:
            continue
        series = frame[col]
        non_empty_rows = int(series.map(_is_non_empty).sum())
        mapped_classes = _column_mapped_classes(col, cfg)
        finish_rows.append(
            {
                "column_name": col,
                "source_column_name": col,
                "normalized_column_name": _normalize_col(col),
                "non_empty_rows": non_empty_rows,
                "non_empty_share": _column_non_empty_share(series),
                "positive_rows": int(series.map(_is_positive).sum()),
                "positive_share": _column_positive_share(series),
                "distinct_non_empty_values": int(_non_empty_value_counts(series, top_k=max(1, len(series))).shape[0]),
                "top_values": json.dumps(_top_values(series), ensure_ascii=False),
                "value_frequencies": _value_frequencies_json(series),
                "covered_by_mapping": bool(mapped_classes),
                "mapped_classes": json.dumps(mapped_classes, ensure_ascii=False),
                "mapping_status": "mapped" if mapped_classes else "unsupported",
            }
        )

    raw_finish_summary = (
        pd.DataFrame(finish_rows)
        if finish_rows
        else pd.DataFrame(
            columns=[
                "column_name",
                "source_column_name",
                "normalized_column_name",
                "non_empty_rows",
                "non_empty_share",
                "positive_rows",
                "positive_share",
                "distinct_non_empty_values",
                "top_values",
                "value_frequencies",
                "covered_by_mapping",
                "mapped_classes",
                "mapping_status",
            ]
        )
    )
    if not raw_finish_summary.empty:
        raw_finish_summary = raw_finish_summary.sort_values(
            ["positive_rows", "non_empty_rows", "source_column_name"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    unsupported_finish_columns = []
    if not raw_finish_summary.empty:
        unsupported_finish_columns = (
            raw_finish_summary[
                (raw_finish_summary["positive_rows"] > 0)
                & (~raw_finish_summary["covered_by_mapping"].astype(bool))
            ]["source_column_name"]
            .astype(str)
            .tolist()
        )

    unsupported_finish_rows: list[dict[str, Any]] = []
    rows_total = max(1, int(len(frame)))
    for col in unsupported_finish_columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        non_empty = series[series.map(_is_non_empty)].astype(str).str.strip()
        if non_empty.empty:
            continue
        counts = non_empty.value_counts().head(20)
        non_empty_rows = int(non_empty.shape[0])
        positive_rows = int(series.map(_is_positive).sum())
        for value, count in counts.items():
            unsupported_finish_rows.append(
                {
                    "source_column_name": str(col),
                    "normalized_column_name": _normalize_col(col),
                    "raw_value": str(value),
                    "rows": int(count),
                    "share_within_column": float(int(count) / max(1, non_empty_rows)),
                    "share_of_total_rows": float(int(count) / float(rows_total)),
                    "non_empty_rows_in_column": non_empty_rows,
                    "positive_rows_in_column": positive_rows,
                }
            )

    unsupported_finish_values = (
        pd.DataFrame(unsupported_finish_rows)
        if unsupported_finish_rows
        else pd.DataFrame(
            columns=[
                "source_column_name",
                "normalized_column_name",
                "raw_value",
                "rows",
                "share_within_column",
                "share_of_total_rows",
                "non_empty_rows_in_column",
                "positive_rows_in_column",
            ]
        )
    )
    if not unsupported_finish_values.empty:
        unsupported_finish_values = unsupported_finish_values.sort_values(
            ["rows", "source_column_name", "raw_value"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    unsupported_score_values = pd.DataFrame(columns=["score_rounded", "rows", "share"])
    score_values_seen: list[int] = []
    if score_col and score_col in frame.columns:
        numeric = pd.to_numeric(frame[score_col], errors="coerce")
        rounded = numeric.round().dropna().astype(int)
        score_values_seen = sorted(rounded.unique().tolist())
        counts = rounded.value_counts().sort_index()
        unsupported_values = [
            int(v)
            for v in counts.index.tolist()
            if int(v) != 0 and int(v) not in cfg.score_to_class
        ]
        if unsupported_values:
            rows = []
            total = max(1, int(len(rounded)))
            for value in unsupported_values:
                cnt = int(counts.get(value, 0))
                rows.append(
                    {
                        "score_rounded": int(value),
                        "rows": cnt,
                        "share": float(cnt / total),
                    }
                )
            unsupported_score_values = pd.DataFrame(rows).sort_values(
                "rows",
                ascending=False,
            )

    resolution_share = (
        obs.get("observation_resolution_type", pd.Series(dtype="object"))
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )

    mapping_crosstab = _make_mapping_crosstab(obs)

    direct_finish_positive_rows = 0
    if supported_finish_columns:
        positive_matrix = pd.DataFrame(
            {
                col: frame[col].map(_is_positive)
                for col in supported_finish_columns
                if col in frame.columns
            }
        )
        if not positive_matrix.empty:
            direct_finish_positive_rows = int(positive_matrix.any(axis=1).sum())

    unsupported_finish_positive_rows = 0
    if unsupported_finish_columns:
        unsupported_matrix = pd.DataFrame(
            {
                col: frame[col].map(_is_positive)
                for col in unsupported_finish_columns
                if col in frame.columns
            }
        )
        if not unsupported_matrix.empty:
            unsupported_finish_positive_rows = int(unsupported_matrix.any(axis=1).sum())

    supported_finish_positive_columns: list[str] = []
    for col in sorted(supported_finish_columns):
        if col in frame.columns and int(frame[col].map(_is_positive).sum()) > 0:
            supported_finish_positive_columns.append(str(col))

    finish_signal_presence: dict[str, dict[str, Any]] = {}
    for target_class in ("hold", "arm_submission", "leg_submission"):
        class_columns = []
        for col in supported_finish_columns:
            if target_class in _column_mapped_classes(col, cfg):
                class_columns.append(col)

        positive_rows = 0
        if class_columns:
            matrix = pd.DataFrame(
                {
                    col: frame[col].map(_is_positive)
                    for col in class_columns
                    if col in frame.columns
                }
            )
            if not matrix.empty:
                positive_rows = int(matrix.any(axis=1).sum())
        finish_signal_presence[target_class] = {
            "columns": sorted(class_columns),
            "positive_rows": positive_rows,
            "present_in_raw": bool(positive_rows > 0),
        }

    rows_total_int = int(len(frame))
    columns_used = sorted(
        [c for c in [score_col] if c]
        + [c for c in sorted(supported_finish_columns)]
    )
    column_non_empty = {
        col: _column_non_empty_share(frame[col])
        for col in columns_used
        if col in frame.columns
    }
    column_positive = {
        col: _column_positive_share(frame[col])
        for col in columns_used
        if col in frame.columns
    }

    direct_match_class_counts, direct_match_column_counts = _direct_finish_match_counts(obs)
    mapping_gap_detected = bool(
        unsupported_finish_positive_rows > 0
        or not unsupported_finish_values.empty
    )

    summary: dict[str, Any] = {
        "rows_total": rows_total_int,
        "mapping_version": str(cfg.version),
        "score_column_used": score_col,
        "raw_columns_used_for_mapping": columns_used,
        "column_non_empty_share": column_non_empty,
        "column_positive_share": column_positive,
        "resolution_type_share": {str(k): float(v) for k, v in resolution_share.items()},
        "candidate_finish_columns": finish_candidates,
        "candidate_finish_columns_count": int(len(finish_candidates)),
        "supported_finish_columns": sorted(supported_finish_columns),
        "supported_finish_columns_with_positive_values": supported_finish_positive_columns,
        "direct_finish_signal_columns_available": bool(supported_finish_columns),
        "direct_finish_positive_rows": int(direct_finish_positive_rows),
        "direct_finish_positive_share": float(direct_finish_positive_rows / max(1, rows_total_int)),
        "mapped_finish_positive_rows": int(direct_finish_positive_rows),
        "unmapped_finish_positive_rows": int(unsupported_finish_positive_rows),
        "mapping_gap_detected": mapping_gap_detected,
        "unsupported_finish_columns": [
            str(col)
            for col in finish_candidates
            if col not in supported_finish_columns
        ],
        "unsupported_finish_columns_with_positive_values": unsupported_finish_columns,
        "unsupported_finish_values_rows": int(len(unsupported_finish_values)),
        "direct_finish_match_class_counts": direct_match_class_counts,
        "direct_finish_match_column_counts": direct_match_column_counts,
        "score_values_seen": score_values_seen,
        "unsupported_score_values": (
            []
            if unsupported_score_values.empty
            else unsupported_score_values["score_rounded"].astype(int).tolist()
        ),
        "finish_signal_presence": finish_signal_presence,
    }

    warnings: list[str] = []
    if not supported_finish_columns:
        warnings.append(
            "Direct finish columns were not detected in cleaned source columns; observed mapping relies on score fallback."
        )
    elif direct_finish_positive_rows == 0:
        warnings.append(
            "Direct finish columns exist but contain no positive signals in this run."
        )
    if unsupported_finish_columns:
        warnings.append(
            "Some raw finish/action columns contain positive values but are not covered by mapping rules."
        )
    if bool(supported_finish_columns) and mapping_gap_detected:
        warnings.append(
            "Direct finish-like signals are present, but mapping coverage is incomplete (mapping gap detected)."
        )
    if not unsupported_finish_values.empty:
        warnings.append(
            "Unsupported raw finish/action values were detected; review unsupported_finish_values.csv and mapping rules."
        )
    if not unsupported_score_values.empty:
        warnings.append(
            "Unsupported non-zero score values were found; these rows map to unknown without direct finish signals."
        )
    summary["warnings"] = warnings
    summary["direct_finish_observations_available"] = bool(
        supported_finish_columns and direct_finish_positive_rows > 0
    )

    return ObservationAuditResult(
        summary=summary,
        mapping_crosstab=mapping_crosstab,
        raw_finish_signal_summary=raw_finish_summary,
        unsupported_finish_values=unsupported_finish_values.reset_index(drop=True),
        unsupported_score_values=unsupported_score_values.reset_index(drop=True),
    )


def write_observation_audit(
    result: ObservationAuditResult,
    diagnostics_dir: str | Path,
) -> dict[str, str]:
    out = Path(diagnostics_dir)
    out.mkdir(parents=True, exist_ok=True)

    observation_audit_path = out / "observation_audit.json"
    mapping_crosstab_path = out / "observation_mapping_crosstab.csv"
    raw_finish_path = out / "raw_finish_signal_summary.csv"
    unsupported_finish_path = out / "unsupported_finish_values.csv"
    unsupported_score_path = out / "unsupported_score_values.csv"

    observation_audit_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.mapping_crosstab.to_csv(mapping_crosstab_path, index=False)
    result.raw_finish_signal_summary.to_csv(raw_finish_path, index=False)
    result.unsupported_finish_values.to_csv(unsupported_finish_path, index=False)
    result.unsupported_score_values.to_csv(unsupported_score_path, index=False)

    return {
        "observation_audit_json": str(observation_audit_path),
        "observation_mapping_crosstab_csv": str(mapping_crosstab_path),
        "raw_finish_signal_summary_csv": str(raw_finish_path),
        "unsupported_finish_values_csv": str(unsupported_finish_path),
        "unsupported_score_values_csv": str(unsupported_score_path),
    }
