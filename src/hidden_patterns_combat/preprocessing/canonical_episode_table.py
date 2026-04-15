from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

import pandas as pd


_TOTAL_ROW_TOKENS: tuple[str, ...] = ("итог", "сумма", "всего", "total", "summary")


@dataclass(frozen=True)
class CanonicalEpisodeConfig:
    athlete_name_candidates: tuple[str, ...] = (
        "metadata__athlete_name",
        "athlete_name",
        "фио борца",
    )
    athlete_id_candidates: tuple[str, ...] = (
        "metadata__athlete_id",
        "athlete_id",
        "id_athlete",
    )
    opponent_candidates: tuple[str, ...] = (
        "metadata__opponent_name",
        "metadata__opponent",
        "opponent_name",
        "opponent",
        "соперник",
    )
    sheet_candidates: tuple[str, ...] = (
        "metadata__sheet",
        "source_sheet",
        "_sheet",
        "sheet",
    )
    weight_class_candidates: tuple[str, ...] = (
        "metadata__weight_class",
        "weight_category",
        "weight_class",
        "весовая категория",
        "вес",
    )
    tournament_candidates: tuple[str, ...] = (
        "metadata__tournament",
        "metadata__event",
        "metadata__competition",
        "tournament",
        "event",
        "турнир",
    )
    date_candidates: tuple[str, ...] = (
        "metadata__event_date",
        "metadata__date",
        "event_date",
        "date",
        "дата",
    )
    sequence_id_candidates: tuple[str, ...] = (
        "metadata__sequence_id",
        "sequence_id",
        "metadata__bout_id",
        "metadata__match_id",
        "metadata__fight_id",
        "bout_id",
        "bout",
        "match_id",
        "match",
        "fight_id",
        "fight",
        "схватка",
        "поединок",
    )
    episode_id_candidates: tuple[str, ...] = (
        "metadata__episode_id",
        "episode_id",
        "metadata__episode_order",
        "episode_order",
        "metadata__episode_attr_01_02",
        "metadata__episode_attr_01",
        "номер эпизода",
    )
    episode_time_candidates: tuple[str, ...] = (
        "metadata__episode_duration",
        "episode_duration",
        "metadata__episode_time",
        "episode_time",
        "metadata__episode_attr_02_03",
        "metadata__time_attr_01_02",
        "duration",
        "время эпизода",
    )
    pause_time_candidates: tuple[str, ...] = (
        "metadata__pause_duration",
        "pause_duration",
        "metadata__pause_time",
        "pause_time",
        "metadata__episode_attr_03_04",
        "metadata__time_attr_02_03",
        "pause",
        "время паузы",
    )
    score_candidates: tuple[str, ...] = (
        "outcomes__score",
        "observed_result",
        "баллы",
        "score",
    )


@dataclass
class CanonicalEpisodeBuildResult:
    canonical_table: pd.DataFrame
    extraction_info: dict[str, Any]


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        low = candidate.lower()
        if low in lowered:
            return lowered[low]
        for col in df.columns:
            if low in col.lower():
                return col
    return None


def _safe_text(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series([""] * len(index), index=index)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )


def _safe_numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series([0.0] * len(index), index=index)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _is_episode_id_like(value: object) -> bool:
    text = str(value).strip()
    if not text:
        return False
    num = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(num):
        return True
    return bool(re.fullmatch(r"[a-zA-Zа-яА-Я]*\d+[a-zA-Zа-яА-Я]*", text))


def _episode_numeric_id(value: object) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return float(numeric)
    match = re.search(r"(\d+)", text)
    if match:
        return float(match.group(1))
    return None


def _build_surrogate_athlete_id(athlete_name: str, sheet_name: str) -> str:
    key = f"{sheet_name}::{athlete_name}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return f"ath_{digest}"


def _is_total_row(texts: list[str]) -> bool:
    for text in texts:
        low = str(text).strip().lower()
        if not low:
            continue
        if any(token in low for token in _TOTAL_ROW_TOKENS):
            return True
    return False


def _hidden_col(hidden_features: pd.DataFrame | None, name: str, index: pd.Index) -> pd.Series:
    if hidden_features is None or name not in hidden_features.columns:
        return pd.Series([0.0] * len(index), index=index)
    return _safe_numeric(hidden_features[name], index=index)


def _non_empty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})


def _episode_attr_position(col_name: str) -> int | None:
    match = re.search(r"metadata__episode_attr_(\d+)(?:_\d+)?", str(col_name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _sorted_episode_attr_columns(frame: pd.DataFrame) -> list[str]:
    cols = [str(col) for col in frame.columns if str(col).startswith("metadata__episode_attr_")]
    return sorted(
        cols,
        key=lambda col: (_episode_attr_position(col) is None, _episode_attr_position(col) or 10**6, col),
    )


def _next_episode_attr_column(frame: pd.DataFrame, base_col: str | None, step: int) -> str | None:
    if not base_col:
        return None
    attrs = _sorted_episode_attr_columns(frame)
    if not attrs or base_col not in attrs:
        return None
    idx = attrs.index(base_col)
    target = idx + step
    if 0 <= target < len(attrs):
        return attrs[target]
    return None


def _detect_weight_column_from_metadata(
    frame: pd.DataFrame,
    *,
    excluded: set[str],
) -> str | None:
    candidates = [
        str(col)
        for col in frame.columns
        if str(col).startswith("metadata__") and str(col) not in excluded
    ]
    best_col: str | None = None
    best_score = -1.0
    for col in candidates:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            continue
        unique = int(valid.round().nunique())
        if unique <= 1:
            continue
        within_range = float(((valid >= 35) & (valid <= 200)).mean())
        if within_range < 0.80:
            continue
        numeric_share = float(numeric.notna().mean())
        score = numeric_share * 2.0 + within_range - min(unique / 100.0, 0.5)
        if score > best_score and unique <= max(20, int(len(valid) * 0.10)):
            best_col = col
            best_score = score
    return best_col


def _resolve_sequence_id(
    out: pd.DataFrame,
    explicit_series: pd.Series,
    opponent_series: pd.Series,
    tournament_series: pd.Series,
    date_series: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    n_rows = len(out)
    sequence_id = pd.Series([""] * n_rows, index=out.index, dtype=object)
    sequence_quality = pd.Series(["low"] * n_rows, index=out.index, dtype=object)
    sequence_resolution = pd.Series(["surrogate"] * n_rows, index=out.index, dtype=object)
    sequence_reason = pd.Series(["unresolved"] * n_rows, index=out.index, dtype=object)

    explicit = _non_empty(explicit_series)
    explicit_mask = explicit.ne("")
    if explicit_mask.any():
        normalized = (
            out["sheet_name"].astype(str)
            + "::explicit::"
            + explicit.where(explicit.ne(""), "missing")
        )
        sequence_id = sequence_id.where(~explicit_mask, normalized)
        sequence_quality = sequence_quality.where(~explicit_mask, "high")
        sequence_resolution = sequence_resolution.where(~explicit_mask, "explicit")
        sequence_reason = sequence_reason.where(~explicit_mask, "explicit_sequence_id")

    unresolved_idx = sequence_id[sequence_id == ""].index.tolist()
    if not unresolved_idx:
        return sequence_id, sequence_quality, sequence_resolution, sequence_reason

    base_df = out.loc[unresolved_idx, ["sheet_name", "athlete_id", "weight_class", "episode_id", "source_row_index"]].copy()
    base_df["opponent_name"] = _non_empty(opponent_series.loc[unresolved_idx])
    base_df["tournament_name"] = _non_empty(tournament_series.loc[unresolved_idx])
    base_df["event_date"] = _non_empty(date_series.loc[unresolved_idx])

    base_df["context_count"] = (
        base_df["weight_class"].astype(str).str.strip().ne("").astype(int)
        + base_df["opponent_name"].ne("").astype(int)
        + base_df["tournament_name"].ne("").astype(int)
        + base_df["event_date"].ne("").astype(int)
    )

    base_df["base_key"] = (
        base_df["sheet_name"].astype(str)
        + "::"
        + base_df["athlete_id"].astype(str)
        + "::w="
        + base_df["weight_class"].astype(str)
        + "::opp="
        + base_df["opponent_name"].astype(str)
        + "::tour="
        + base_df["tournament_name"].astype(str)
        + "::date="
        + base_df["event_date"].astype(str)
    )

    episode_num = base_df["episode_id"].map(_episode_numeric_id)
    base_df = base_df.assign(_episode_num=episode_num)

    for base_key, group in base_df.groupby("base_key", sort=False):
        ordered = group.sort_values("source_row_index")
        block_idx = 0
        prev_episode_num: float | None = None
        has_numeric = ordered["_episode_num"].notna().any()

        for row_index, row in ordered.iterrows():
            curr_episode_num = row["_episode_num"]
            if prev_episode_num is not None and curr_episode_num is not None and float(curr_episode_num) <= float(prev_episode_num):
                block_idx += 1
            if curr_episode_num is not None:
                prev_episode_num = float(curr_episode_num)

            seq_value = f"{base_key}::block_{block_idx:02d}"
            sequence_id.loc[row_index] = seq_value

            context_count = int(row["context_count"])
            if context_count >= 2 and has_numeric:
                quality = "high"
                reason = "surrogate_context_and_episode_id"
            elif context_count >= 1 or has_numeric:
                quality = "medium"
                reason = "surrogate_episode_only" if context_count == 0 else "surrogate_partial_context"
            else:
                quality = "low"
                reason = "surrogate_no_context"

            sequence_quality.loc[row_index] = quality
            sequence_resolution.loc[row_index] = "surrogate"
            sequence_reason.loc[row_index] = reason

    # Deterministic fallback if something is still empty for any reason.
    empty_mask = sequence_id.eq("")
    if empty_mask.any():
        fallback = (
            out.loc[empty_mask, "sheet_name"].astype(str)
            + "::fallback::"
            + out.loc[empty_mask, "athlete_id"].astype(str)
            + "::"
            + out.loc[empty_mask, "source_row_index"].astype(str)
        )
        sequence_id.loc[empty_mask] = fallback
        sequence_quality.loc[empty_mask] = "low"
        sequence_resolution.loc[empty_mask] = "fallback"
        sequence_reason.loc[empty_mask] = "fallback_source_row_index"

    return sequence_id, sequence_quality, sequence_resolution, sequence_reason


def build_canonical_episode_table(
    cleaned_df: pd.DataFrame,
    observation_df: pd.DataFrame,
    hidden_features: pd.DataFrame | None = None,
    config: CanonicalEpisodeConfig | None = None,
) -> CanonicalEpisodeBuildResult:
    cfg = config or CanonicalEpisodeConfig()
    frame = cleaned_df.copy().reset_index(drop=True)
    obs = observation_df.copy().reset_index(drop=True)

    if len(frame) != len(obs):
        raise ValueError(
            "canonical episode table requires aligned cleaned and observation rows: "
            f"cleaned={len(frame)} observations={len(obs)}"
        )

    selection_method: dict[str, str] = {}

    athlete_col = _first_existing(frame, cfg.athlete_name_candidates)
    selection_method["athlete_name"] = "candidate_match" if athlete_col else "missing"

    athlete_id_col = _first_existing(frame, cfg.athlete_id_candidates)
    selection_method["athlete_id"] = "candidate_match" if athlete_id_col else "surrogate_from_athlete_name"

    opponent_col = _first_existing(frame, cfg.opponent_candidates)
    selection_method["opponent_name"] = "candidate_match" if opponent_col else "missing"

    sheet_col = _first_existing(frame, cfg.sheet_candidates)
    selection_method["sheet_name"] = "candidate_match" if sheet_col else "missing"

    weight_col = _first_existing(frame, cfg.weight_class_candidates)
    selection_method["weight_class"] = "candidate_match" if weight_col else "missing"

    tournament_col = _first_existing(frame, cfg.tournament_candidates)
    selection_method["tournament_name"] = "candidate_match" if tournament_col else "missing"

    date_col = _first_existing(frame, cfg.date_candidates)
    selection_method["event_date"] = "candidate_match" if date_col else "missing"

    explicit_sequence_col = _first_existing(frame, cfg.sequence_id_candidates)
    selection_method["explicit_sequence_id"] = "candidate_match" if explicit_sequence_col else "missing"

    episode_col = _first_existing(frame, cfg.episode_id_candidates)
    if episode_col:
        selection_method["episode_id"] = "candidate_match"
    else:
        episode_attrs = _sorted_episode_attr_columns(frame)
        if episode_attrs:
            episode_col = episode_attrs[0]
            selection_method["episode_id"] = "positional_episode_attr_fallback"
        else:
            selection_method["episode_id"] = "missing"

    episode_time_col = _first_existing(frame, cfg.episode_time_candidates)
    if episode_time_col:
        selection_method["episode_time_sec"] = "candidate_match"
    else:
        fallback_episode_time = _next_episode_attr_column(frame, base_col=episode_col, step=1)
        if fallback_episode_time:
            episode_time_col = fallback_episode_time
            selection_method["episode_time_sec"] = "positional_after_episode_id"
        else:
            selection_method["episode_time_sec"] = "missing"

    pause_col = _first_existing(frame, cfg.pause_time_candidates)
    if pause_col:
        selection_method["pause_time_sec"] = "candidate_match"
    else:
        fallback_pause = _next_episode_attr_column(frame, base_col=episode_col, step=2)
        if not fallback_pause and episode_time_col:
            fallback_pause = _next_episode_attr_column(frame, base_col=episode_time_col, step=1)
        if fallback_pause:
            pause_col = fallback_pause
            selection_method["pause_time_sec"] = "positional_after_episode_time"
        else:
            selection_method["pause_time_sec"] = "missing"

    score_col = _first_existing(frame, cfg.score_candidates)
    selection_method["score"] = "candidate_match" if score_col else "missing"

    if not weight_col:
        guessed_weight = _detect_weight_column_from_metadata(
            frame,
            excluded={c for c in [episode_col, episode_time_col, pause_col] if c},
        )
        if guessed_weight:
            weight_col = guessed_weight
            selection_method["weight_class"] = "heuristic_metadata_numeric_range"

    out = pd.DataFrame(index=frame.index)

    out["source_row_index"] = frame.index.astype(int)
    out["sheet_name"] = _safe_text(frame[sheet_col] if sheet_col else None, index=frame.index)

    athlete_name = _safe_text(frame[athlete_col] if athlete_col else None, index=frame.index)
    out["athlete_name"] = athlete_name

    athlete_id_raw = _safe_text(frame[athlete_id_col] if athlete_id_col else None, index=frame.index)
    out["athlete_id"] = [
        athlete_id_raw.iloc[i] if athlete_id_raw.iloc[i] else _build_surrogate_athlete_id(athlete_name.iloc[i], out["sheet_name"].iloc[i])
        for i in range(len(out))
    ]

    out["weight_class"] = _safe_text(frame[weight_col] if weight_col else None, index=frame.index)
    out["episode_id"] = _safe_text(frame[episode_col] if episode_col else None, index=frame.index)
    out["opponent_name"] = _safe_text(frame[opponent_col] if opponent_col else None, index=frame.index)
    out["tournament_name"] = _safe_text(frame[tournament_col] if tournament_col else None, index=frame.index)
    out["event_date"] = _safe_text(frame[date_col] if date_col else None, index=frame.index)
    out["episode_time_sec"] = _safe_numeric(frame[episode_time_col] if episode_time_col else None, index=frame.index)
    out["pause_time_sec"] = _safe_numeric(frame[pause_col] if pause_col else None, index=frame.index)
    out["score"] = _safe_numeric(frame[score_col] if score_col else None, index=frame.index)

    out["maneuver_right_code"] = _hidden_col(hidden_features, "maneuver_right_code", out.index)
    out["maneuver_left_code"] = _hidden_col(hidden_features, "maneuver_left_code", out.index)
    out["kfv_capture_code"] = _hidden_col(hidden_features, "grips_code", out.index)
    out["kfv_grip_code"] = _hidden_col(hidden_features, "holds_code", out.index)
    out["kfv_wrap_code"] = _hidden_col(hidden_features, "bodylocks_code", out.index)
    out["kfv_hook_code"] = _hidden_col(hidden_features, "underhooks_code", out.index)
    out["kfv_post_code"] = _hidden_col(hidden_features, "posts_code", out.index)
    out["vup_code"] = _hidden_col(hidden_features, "vup_code", out.index)

    out["observed_zap_class"] = obs.get("observed_zap_class", pd.Series(["unknown"] * len(out))).astype(str)
    out["observed_zap_source_columns"] = obs.get(
        "observed_zap_source_columns",
        pd.Series(["[]"] * len(out)),
    ).astype(str)
    out["observation_quality_flag"] = obs.get(
        "observation_quality_flag",
        pd.Series(["unknown_missing_observation"] * len(out)),
    ).astype(str)
    out["observation_resolution_type"] = obs.get(
        "observation_resolution_type",
        pd.Series(["unknown"] * len(out)),
    ).astype(str)
    out["observation_confidence_label"] = obs.get(
        "observation_confidence_label",
        pd.Series(["low"] * len(out)),
    ).astype(str)
    out["mapping_version"] = obs.get("mapping_version", pd.Series(["unknown"] * len(out))).astype(str)
    out["finish_match_classes"] = obs.get(
        "finish_match_classes",
        pd.Series(["[]"] * len(out)),
    ).astype(str)
    out["finish_match_columns"] = obs.get(
        "finish_match_columns",
        pd.Series(["[]"] * len(out)),
    ).astype(str)
    out["score_value"] = pd.to_numeric(
        obs.get("score_value", pd.Series([None] * len(out))),
        errors="coerce",
    )
    out["score_rounded"] = pd.to_numeric(
        obs.get("score_rounded", pd.Series([None] * len(out))),
        errors="coerce",
    )
    out["score_supported_class"] = obs.get(
        "score_supported_class",
        pd.Series([""] * len(out)),
    ).astype(str)

    explicit_series = _safe_text(frame[explicit_sequence_col] if explicit_sequence_col else None, index=frame.index)
    opponent_series = out["opponent_name"].copy()
    tournament_series = out["tournament_name"].copy()
    date_series = out["event_date"].copy()
    sequence_id, sequence_quality, sequence_resolution, sequence_reason = _resolve_sequence_id(
        out=out,
        explicit_series=explicit_series,
        opponent_series=opponent_series,
        tournament_series=tournament_series,
        date_series=date_series,
    )
    out["sequence_id"] = sequence_id.astype(str)
    out["sequence_quality_flag"] = sequence_quality.astype(str)
    out["sequence_resolution_type"] = sequence_resolution.astype(str)
    out["sequence_quality_reason"] = sequence_reason.astype(str)

    total_flags: list[bool] = []
    for i in range(len(out)):
        texts = [
            out["athlete_name"].iloc[i],
            out["episode_id"].iloc[i],
            out["weight_class"].iloc[i],
        ]
        total_flags.append(_is_total_row(texts))
    out["is_total_row"] = pd.Series(total_flags, index=out.index)

    has_episode_id = out["episode_id"].map(_is_episode_id_like)
    out["is_train_eligible"] = (
        (~out["is_total_row"])
        & has_episode_id
        & out["observed_zap_class"].ne("unknown")
        & out["sequence_quality_flag"].isin(["high", "medium"])
    )

    out["source_record_id"] = out.apply(
        lambda row: f"{row['sheet_name']}::{int(row['source_row_index'])}",
        axis=1,
    )

    ordered_cols = [
        "athlete_name",
        "athlete_id",
        "sheet_name",
        "weight_class",
        "opponent_name",
        "tournament_name",
        "event_date",
        "episode_id",
        "sequence_id",
        "sequence_quality_flag",
        "sequence_resolution_type",
        "sequence_quality_reason",
        "episode_time_sec",
        "pause_time_sec",
        "score",
        "score_value",
        "score_rounded",
        "score_supported_class",
        "maneuver_right_code",
        "maneuver_left_code",
        "kfv_capture_code",
        "kfv_grip_code",
        "kfv_wrap_code",
        "kfv_hook_code",
        "kfv_post_code",
        "vup_code",
        "observed_zap_class",
        "observed_zap_source_columns",
        "finish_match_classes",
        "finish_match_columns",
        "observation_quality_flag",
        "observation_resolution_type",
        "observation_confidence_label",
        "mapping_version",
        "is_total_row",
        "is_train_eligible",
        "source_row_index",
        "source_record_id",
    ]
    out = out[ordered_cols]

    extraction_info: dict[str, Any] = {
        "selected_columns": {
            "athlete_name": athlete_col,
            "athlete_id": athlete_id_col,
            "sheet_name": sheet_col,
            "weight_class": weight_col,
            "opponent_name": opponent_col,
            "tournament_name": tournament_col,
            "event_date": date_col,
            "explicit_sequence_id": explicit_sequence_col,
            "episode_id": episode_col,
            "episode_time_sec": episode_time_col,
            "pause_time_sec": pause_col,
            "score": score_col,
        },
        "selection_method": selection_method,
    }

    return CanonicalEpisodeBuildResult(canonical_table=out, extraction_info=extraction_info)
