from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

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
        "opponent_name",
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
        "weight_class",
        "весовая категория",
        "вес",
    )
    tournament_candidates: tuple[str, ...] = (
        "metadata__tournament",
        "metadata__event",
        "tournament",
        "event",
        "турнир",
    )
    date_candidates: tuple[str, ...] = (
        "metadata__event_date",
        "event_date",
        "date",
        "дата",
    )
    sequence_id_candidates: tuple[str, ...] = (
        "metadata__sequence_id",
        "sequence_id",
        "metadata__bout_id",
        "bout_id",
        "match_id",
        "fight_id",
    )
    episode_id_candidates: tuple[str, ...] = (
        "metadata__episode_id",
        "episode_id",
        "metadata__episode_attr_01",
        "номер эпизода",
    )
    episode_time_candidates: tuple[str, ...] = (
        "metadata__episode_duration",
        "episode_duration",
        "duration",
        "время эпизода",
    )
    pause_time_candidates: tuple[str, ...] = (
        "metadata__pause_duration",
        "pause_duration",
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


def _resolve_sequence_id(
    out: pd.DataFrame,
    explicit_series: pd.Series,
    opponent_series: pd.Series,
    tournament_series: pd.Series,
    date_series: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    n_rows = len(out)
    sequence_id = pd.Series([""] * n_rows, index=out.index, dtype=object)
    sequence_quality = pd.Series(["low"] * n_rows, index=out.index, dtype=object)
    sequence_resolution = pd.Series(["surrogate"] * n_rows, index=out.index, dtype=object)

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

    unresolved_idx = sequence_id[sequence_id == ""].index.tolist()
    if not unresolved_idx:
        return sequence_id, sequence_quality, sequence_resolution

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
            elif context_count >= 1 or has_numeric:
                quality = "medium"
            else:
                quality = "low"

            sequence_quality.loc[row_index] = quality
            sequence_resolution.loc[row_index] = "surrogate"

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

    return sequence_id, sequence_quality, sequence_resolution


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

    athlete_col = _first_existing(frame, cfg.athlete_name_candidates)
    athlete_id_col = _first_existing(frame, cfg.athlete_id_candidates)
    opponent_col = _first_existing(frame, cfg.opponent_candidates)
    sheet_col = _first_existing(frame, cfg.sheet_candidates)
    weight_col = _first_existing(frame, cfg.weight_class_candidates)
    tournament_col = _first_existing(frame, cfg.tournament_candidates)
    date_col = _first_existing(frame, cfg.date_candidates)
    explicit_sequence_col = _first_existing(frame, cfg.sequence_id_candidates)
    episode_col = _first_existing(frame, cfg.episode_id_candidates)
    episode_time_col = _first_existing(frame, cfg.episode_time_candidates)
    pause_col = _first_existing(frame, cfg.pause_time_candidates)
    score_col = _first_existing(frame, cfg.score_candidates)

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

    explicit_series = _safe_text(frame[explicit_sequence_col] if explicit_sequence_col else None, index=frame.index)
    opponent_series = _safe_text(frame[opponent_col] if opponent_col else None, index=frame.index)
    tournament_series = _safe_text(frame[tournament_col] if tournament_col else None, index=frame.index)
    date_series = _safe_text(frame[date_col] if date_col else None, index=frame.index)
    sequence_id, sequence_quality, sequence_resolution = _resolve_sequence_id(
        out=out,
        explicit_series=explicit_series,
        opponent_series=opponent_series,
        tournament_series=tournament_series,
        date_series=date_series,
    )
    out["sequence_id"] = sequence_id.astype(str)
    out["sequence_quality_flag"] = sequence_quality.astype(str)
    out["sequence_resolution_type"] = sequence_resolution.astype(str)

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
        "episode_id",
        "sequence_id",
        "sequence_quality_flag",
        "sequence_resolution_type",
        "episode_time_sec",
        "pause_time_sec",
        "score",
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

    return CanonicalEpisodeBuildResult(canonical_table=out)
