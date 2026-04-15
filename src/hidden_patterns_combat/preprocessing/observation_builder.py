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
            observed_class.append("unknown")
            quality.append(finish_error)
            source_cols.append(json.dumps(sorted(set(row_sources)), ensure_ascii=False))
            continue

        if finish_class is not None:
            if score_class is not None and score_class != finish_class:
                observed_class.append(finish_class)
                quality.append("ok_finish_rule_score_mismatch")
            else:
                observed_class.append(finish_class)
                quality.append("ok_finish_rule")
            source_cols.append(json.dumps(sorted(set(row_sources)), ensure_ascii=False))
            continue

        # No finish action is active: explicit, deterministic score rule.
        if score_value is None:
            observed_class.append("unknown")
            quality.append("unknown_missing_score_and_finish")
            source_cols.append(json.dumps(sorted(set(row_sources)), ensure_ascii=False))
            continue

        if abs(score_value) < 1e-12:
            observed_class.append("no_score")
            quality.append("ok_no_score_rule")
            source_cols.append(json.dumps(sorted(set(row_sources)), ensure_ascii=False))
            continue

        if score_class is None:
            observed_class.append("unknown")
            quality.append("unknown_unsupported_score")
            source_cols.append(json.dumps(sorted(set(row_sources)), ensure_ascii=False))
            continue

        observed_class.append(score_class)
        quality.append("ok_score_rule")
        source_cols.append(json.dumps(sorted(set(row_sources)), ensure_ascii=False))

    out = pd.DataFrame(index=frame.index)
    out["observed_zap_class"] = observed_class
    out["observed_zap_source_columns"] = source_cols
    out["observation_quality_flag"] = quality
    out["mapping_version"] = cfg.version

    out["observed_zap_class"] = out["observed_zap_class"].where(
        out["observed_zap_class"].isin(CANONICAL_OBSERVED_CLASSES),
        "unknown",
    )

    return ObservationBuildResult(observations=out)
