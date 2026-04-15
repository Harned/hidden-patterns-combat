from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


CANONICAL_OBSERVED_CLASSES: tuple[str, ...] = (
    "zap_r",
    "zap_n",
    "zap_t",
    "hold",
    "arm_submission",
    "leg_submission",
    "no_score",
    "unknown",
)


OBSERVATION_RESOLUTION_TYPES: tuple[str, ...] = (
    "direct_finish_signal",
    "inferred_from_score",
    "no_score_rule",
    "ambiguous",
    "unknown",
)


OBSERVATION_CONFIDENCE_LABELS: tuple[str, ...] = ("high", "medium", "low")


@dataclass(frozen=True)
class ObservationFinishRule:
    class_name: str
    match_any_tokens: tuple[str, ...]
    match_all_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ObservationMappingConfig:
    version: str
    score_column_candidates: tuple[str, ...]
    score_to_class: dict[int, str]
    finish_rules: tuple[ObservationFinishRule, ...]


@dataclass
class ObservationBuildResult:
    observations: pd.DataFrame


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "observation_mapping_v1.json"


def load_observation_mapping_config(path: str | Path | None = None) -> ObservationMappingConfig:
    cfg_path = Path(path) if path else _default_config_path()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))

    score_to_class: dict[int, str] = {}
    for key, value in (payload.get("score_to_class") or {}).items():
        score_to_class[int(key)] = str(value)

    finish_rules = tuple(
        ObservationFinishRule(
            class_name=str(row.get("class_name", "unknown")),
            match_any_tokens=tuple(str(x).lower() for x in (row.get("match_any_tokens") or [])),
            match_all_tokens=tuple(str(x).lower() for x in (row.get("match_all_tokens") or [])),
        )
        for row in (payload.get("finish_rules") or [])
    )

    return ObservationMappingConfig(
        version=str(payload.get("version", "observation_mapping_v1")),
        score_column_candidates=tuple(str(x) for x in (payload.get("score_column_candidates") or [])),
        score_to_class=score_to_class,
        finish_rules=finish_rules,
    )


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        candidate_low = candidate.lower()
        if candidate_low in lowered:
            return lowered[candidate_low]
        for col in df.columns:
            if candidate_low in col.lower():
                return col
    return None


def _tokenize_column(col: str) -> str:
    return str(col).strip().lower().replace("ё", "е")


def _is_positive_value(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return False
    if text in {"да", "yes", "true", "истина", "y"}:
        return True
    if text in {"нет", "no", "false", "ложь", "n"}:
        return False

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return float(numeric) > 0.0
    return False


def _safe_score(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return None
    numeric = pd.to_numeric(pd.Series([text.replace(",", ".")]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _active_rule_columns(row: pd.Series, rule: ObservationFinishRule) -> list[str]:
    matched_columns: list[str] = []
    for col in row.index:
        normalized_col = _tokenize_column(col)
        all_ok = all(token in normalized_col for token in rule.match_all_tokens) if rule.match_all_tokens else False
        any_ok = any(token in normalized_col for token in rule.match_any_tokens) if rule.match_any_tokens else False
        if rule.match_all_tokens and rule.match_any_tokens:
            matches = all_ok or any_ok
        elif rule.match_all_tokens:
            matches = all_ok
        elif rule.match_any_tokens:
            matches = any_ok
        else:
            matches = False
        if not matches:
            continue
        if _is_positive_value(row[col]):
            matched_columns.append(str(col))
    return matched_columns


def _class_from_finish_rules(
    row: pd.Series,
    cfg: ObservationMappingConfig,
) -> tuple[str | None, list[str], str | None]:
    winners: list[tuple[str, str]] = []
    for rule in cfg.finish_rules:
        for col in _active_rule_columns(row, rule):
            winners.append((rule.class_name, col))

    if not winners:
        return None, [], None

    unique_classes = sorted({name for name, _ in winners})
    source_cols = sorted({col for _, col in winners})

    if len(unique_classes) > 1:
        return None, source_cols, "unknown_ambiguous_finish"

    return unique_classes[0], source_cols, None


def _append_row(
    observed_class: list[str],
    source_cols: list[str],
    quality: list[str],
    resolution_type: list[str],
    confidence_label: list[str],
    *,
    class_name: str,
    source: list[str],
    quality_flag: str,
    resolution: str,
    confidence: str,
) -> None:
    observed_class.append(class_name)
    source_cols.append(json.dumps(sorted(set(source)), ensure_ascii=False))
    quality.append(quality_flag)
    resolution_type.append(resolution)
    confidence_label.append(confidence)


def build_observed_zap_classes(
    cleaned_df: pd.DataFrame,
    config: ObservationMappingConfig | None = None,
) -> ObservationBuildResult:
    cfg = config or load_observation_mapping_config()
    frame = cleaned_df.copy()

    score_col = _first_existing_column(frame, cfg.score_column_candidates)

    observed_class: list[str] = []
    source_cols: list[str] = []
    quality: list[str] = []
    resolution_type: list[str] = []
    confidence_label: list[str] = []

    for _, row in frame.iterrows():
        finish_class, finish_sources, finish_error = _class_from_finish_rules(row, cfg)

        score_value = _safe_score(row.get(score_col) if score_col else None)
        score_class = None
        if score_value is not None:
            rounded = int(round(score_value))
            score_class = cfg.score_to_class.get(rounded)

        row_sources = list(finish_sources)
        if score_col:
            row_sources.append(score_col)

        if finish_error:
            _append_row(
                observed_class,
                source_cols,
                quality,
                resolution_type,
                confidence_label,
                class_name="unknown",
                source=row_sources,
                quality_flag=finish_error,
                resolution="ambiguous",
                confidence="low",
            )
            continue

        if finish_class is not None:
            quality_flag = "ok_finish_rule"
            if score_class is not None and score_class != finish_class:
                quality_flag = "ok_finish_rule_score_mismatch"
            _append_row(
                observed_class,
                source_cols,
                quality,
                resolution_type,
                confidence_label,
                class_name=finish_class,
                source=row_sources,
                quality_flag=quality_flag,
                resolution="direct_finish_signal",
                confidence="high",
            )
            continue

        # Controlled fallback: score mapping is secondary and explicit.
        if score_value is None:
            _append_row(
                observed_class,
                source_cols,
                quality,
                resolution_type,
                confidence_label,
                class_name="unknown",
                source=row_sources,
                quality_flag="unknown_missing_score_and_finish",
                resolution="unknown",
                confidence="low",
            )
            continue

        if abs(score_value) < 1e-12:
            _append_row(
                observed_class,
                source_cols,
                quality,
                resolution_type,
                confidence_label,
                class_name="no_score",
                source=row_sources,
                quality_flag="ok_no_score_rule",
                resolution="no_score_rule",
                confidence="high",
            )
            continue

        if score_class is None:
            _append_row(
                observed_class,
                source_cols,
                quality,
                resolution_type,
                confidence_label,
                class_name="unknown",
                source=row_sources,
                quality_flag="unknown_unsupported_score",
                resolution="unknown",
                confidence="low",
            )
            continue

        _append_row(
            observed_class,
            source_cols,
            quality,
            resolution_type,
            confidence_label,
            class_name=score_class,
            source=row_sources,
            quality_flag="ok_score_rule",
            resolution="inferred_from_score",
            confidence="medium",
        )

    out = pd.DataFrame(index=frame.index)
    out["observed_zap_class"] = observed_class
    out["observed_zap_source_columns"] = source_cols
    out["observation_quality_flag"] = quality
    out["observation_resolution_type"] = resolution_type
    out["observation_confidence_label"] = confidence_label
    out["mapping_version"] = cfg.version

    out["observed_zap_class"] = out["observed_zap_class"].where(
        out["observed_zap_class"].isin(CANONICAL_OBSERVED_CLASSES),
        "unknown",
    )
    out["observation_resolution_type"] = out["observation_resolution_type"].where(
        out["observation_resolution_type"].isin(OBSERVATION_RESOLUTION_TYPES),
        "unknown",
    )
    out["observation_confidence_label"] = out["observation_confidence_label"].where(
        out["observation_confidence_label"].isin(OBSERVATION_CONFIDENCE_LABELS),
        "low",
    )

    return ObservationBuildResult(observations=out)
