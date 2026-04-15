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
    sheet_col = _first_existing(frame, cfg.sheet_candidates)
    weight_col = _first_existing(frame, cfg.weight_class_candidates)
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
    out["mapping_version"] = obs.get("mapping_version", pd.Series(["unknown"] * len(out))).astype(str)

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
        "mapping_version",
        "is_total_row",
        "is_train_eligible",
        "source_row_index",
        "source_record_id",
    ]
    out = out[ordered_cols]

    return CanonicalEpisodeBuildResult(canonical_table=out)
