from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-9
_ANCHOR_POWER = 1.6
_ANCHOR_BLOCK_WEIGHTS = {
    "s1": 0.85,  # maneuvering is usually dense; keep but reduce bias.
    "s2": 1.00,
    "s3": 1.25,  # VUP tends to be sparse; mildly upweight for identifiability.
}


HIDDEN_STATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "maneuver_right_code",
    "maneuver_left_code",
    "kfv_capture_code",
    "kfv_grip_code",
    "kfv_wrap_code",
    "kfv_hook_code",
    "kfv_post_code",
    "vup_code",
    "episode_time_sec",
    "pause_time_sec",
    "duration_bin",
    "pause_bin",
    "sequence_progress",
    "anchor_s1",
    "anchor_s2",
    "anchor_s3",
    "train_weight",
)


@dataclass
class HiddenStateFeatureLayer:
    hidden_state_features: pd.DataFrame
    observed_sequence: pd.Series


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _normalized_sequence_ids(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.get("sequence_id", pd.Series(["sequence_0"] * len(frame), index=frame.index))
        .fillna("sequence_0")
        .astype(str)
        .str.strip()
        .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
    )


def _safe_text_series(frame: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype="object")
    return (
        frame[column]
        .fillna(default)
        .astype(str)
        .str.strip()
        .replace({"": default, "nan": default, "None": default})
    )


def _minmax(series: pd.Series) -> pd.Series:
    values = _safe_numeric_series(series)
    vmin = float(values.min())
    vmax = float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return (values - vmin) / (vmax - vmin)


def _sequence_progress(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    sequence_ids = _normalized_sequence_ids(frame)
    source_row = pd.to_numeric(
        frame.get("source_row_index", pd.Series(np.arange(len(frame)), index=frame.index)),
        errors="coerce",
    )
    if source_row.isna().any():
        fallback = pd.Series(np.arange(len(frame)), index=frame.index, dtype=float)
        source_row = source_row.where(source_row.notna(), fallback)
    source_row = source_row.astype(float)

    progress = pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
    order_frame = pd.DataFrame(
        {
            "_sequence_id": sequence_ids,
            "_source_row_index": source_row,
        },
        index=frame.index,
    )

    for _, group in order_frame.groupby("_sequence_id", sort=False):
        ordered = group.sort_values("_source_row_index")
        n_rows = len(ordered)
        if n_rows <= 1:
            progress.loc[ordered.index] = 0.0
            continue
        progress.loc[ordered.index] = np.linspace(0.0, 1.0, n_rows)

    return progress.astype(float)


def _quantile_bin(series: pd.Series, *, q_low: float = 0.33, q_high: float = 0.66) -> pd.Series:
    values = _safe_numeric_series(series)
    positive = values[values > 0]
    if positive.empty:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)

    lo = float(positive.quantile(q_low))
    hi = float(positive.quantile(q_high))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo
    if hi <= lo:
        hi = lo + 1e-6

    out = pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    out = out.where(values <= lo, 1.0)
    out = out.where(values <= hi, 2.0)
    return out


def _relative_anchor_components(
    maneuver_total: pd.Series,
    kfv_total: pd.Series,
    vup_total: pd.Series,
) -> pd.DataFrame:
    block = pd.DataFrame(
        {
            "s1": _minmax(maneuver_total).clip(lower=0.0) * _ANCHOR_BLOCK_WEIGHTS["s1"],
            "s2": _minmax(kfv_total).clip(lower=0.0) * _ANCHOR_BLOCK_WEIGHTS["s2"],
            "s3": _minmax(vup_total).clip(lower=0.0) * _ANCHOR_BLOCK_WEIGHTS["s3"],
        },
        index=maneuver_total.index,
    )

    # Relative anchors are more stable than absolute magnitudes when one block
    # (usually maneuvering) has systematically larger numeric range.
    block = block.clip(lower=0.0) + _EPS
    block = block.div(block.sum(axis=1).replace(0.0, 1.0), axis=0)
    block = block.pow(_ANCHOR_POWER)
    block = block.div(block.sum(axis=1).replace(0.0, 1.0), axis=0)
    return block


def _build_train_weight(frame: pd.DataFrame) -> pd.Series:
    observed = _safe_text_series(frame, "observed_zap_class", "unknown").str.lower()
    resolution = _safe_text_series(frame, "observation_resolution_type", "unknown").str.lower()
    confidence = _safe_text_series(frame, "observation_confidence_label", "low").str.lower()
    sequence_quality = _safe_text_series(frame, "sequence_quality_flag", "low").str.lower()

    train_eligible = (
        pd.to_numeric(frame.get("is_train_eligible", pd.Series([1.0] * len(frame), index=frame.index)), errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    train_eligible = train_eligible > 0.5

    weights = pd.Series(np.ones(len(frame), dtype=float), index=frame.index)

    observed_weight_map = {
        "no_score": 0.12,
        "unknown": 0.015,
    }
    resolution_weight_map = {
        "direct_finish_signal": 1.00,
        "inferred_from_score": 0.80,
        "no_score_rule": 0.18,
        "ambiguous": 0.02,
        "unknown": 0.01,
    }
    confidence_weight_map = {"high": 1.00, "medium": 0.80, "low": 0.30}
    sequence_weight_map = {"high": 1.00, "medium": 0.90, "low": 0.30}

    weights *= observed.map(observed_weight_map).fillna(1.0)
    weights *= resolution.map(resolution_weight_map).fillna(0.35)
    weights *= confidence.map(confidence_weight_map).fillna(0.35)
    weights *= sequence_quality.map(sequence_weight_map).fillna(0.35)
    weights = weights.where(train_eligible, 0.0)
    return weights.clip(lower=0.0, upper=1.0)


def build_hidden_state_feature_layer(canonical_episode_table: pd.DataFrame) -> HiddenStateFeatureLayer:
    frame = canonical_episode_table.copy().reset_index(drop=True)

    hidden = pd.DataFrame(index=frame.index)
    base_numeric = [
        "maneuver_right_code",
        "maneuver_left_code",
        "kfv_capture_code",
        "kfv_grip_code",
        "kfv_wrap_code",
        "kfv_hook_code",
        "kfv_post_code",
        "vup_code",
        "episode_time_sec",
        "pause_time_sec",
    ]
    for col in base_numeric:
        if col in frame.columns:
            hidden[col] = _safe_numeric_series(frame[col])
        else:
            hidden[col] = 0.0

    maneuver_total = hidden["maneuver_right_code"] + hidden["maneuver_left_code"]
    kfv_total = (
        hidden["kfv_capture_code"]
        + hidden["kfv_grip_code"]
        + hidden["kfv_wrap_code"]
        + hidden["kfv_hook_code"]
        + hidden["kfv_post_code"]
    )
    vup_total = hidden["vup_code"]

    duration_bin = _quantile_bin(hidden["episode_time_sec"])
    pause_bin = _quantile_bin(hidden["pause_time_sec"])
    duration_norm = duration_bin / max(1.0, float(duration_bin.max()))
    pause_norm = pause_bin / max(1.0, float(pause_bin.max()))
    sequence_progress = _sequence_progress(frame)
    mid_phase = (1.0 - (sequence_progress - 0.5).abs() * 2.0).clip(lower=0.0, upper=1.0)
    relative = _relative_anchor_components(
        maneuver_total=maneuver_total,
        kfv_total=kfv_total,
        vup_total=vup_total,
    )
    maneuver_abs = _minmax(maneuver_total)
    kfv_abs = _minmax(kfv_total)
    vup_abs = _minmax(vup_total)

    hidden["duration_bin"] = duration_bin
    hidden["pause_bin"] = pause_bin
    hidden["sequence_progress"] = sequence_progress

    anchor_s1_raw = (
        0.50 * relative["s1"]
        + 0.20 * maneuver_abs
        + 0.20 * (1.0 - sequence_progress)
        + 0.10 * (1.0 - duration_norm)
    )
    anchor_s2_raw = (
        0.45 * relative["s2"]
        + 0.20 * kfv_abs
        + 0.25 * mid_phase
        + 0.10 * duration_norm
    )
    anchor_s3_raw = (
        0.45 * relative["s3"]
        + 0.20 * vup_abs
        + 0.20 * sequence_progress
        + 0.10 * duration_norm
        + 0.05 * (1.0 - pause_norm)
    )
    anchors = pd.DataFrame(
        {
            "anchor_s1": anchor_s1_raw.clip(lower=0.0),
            "anchor_s2": anchor_s2_raw.clip(lower=0.0),
            "anchor_s3": anchor_s3_raw.clip(lower=0.0),
        },
        index=hidden.index,
    )
    anchors = anchors + _EPS
    anchors = anchors.div(anchors.sum(axis=1).replace(0.0, 1.0), axis=0)

    # Low-information rows (mostly no_score/unknown) should not collapse all anchors
    # to maneuvering. Blend them with a simple left-to-right stage prior.
    stage = pd.DataFrame(
        {
            "anchor_s1": (1.0 - sequence_progress).clip(lower=0.0, upper=1.0),
            "anchor_s2": mid_phase,
            "anchor_s3": sequence_progress.clip(lower=0.0, upper=1.0),
        },
        index=hidden.index,
    )
    stage = stage + _EPS
    stage = stage.div(stage.sum(axis=1).replace(0.0, 1.0), axis=0)

    observed_text = _safe_text_series(frame, "observed_zap_class", "unknown").str.lower()
    resolution_text = _safe_text_series(frame, "observation_resolution_type", "unknown").str.lower()
    low_info_mask = observed_text.isin({"no_score", "unknown"}) | resolution_text.isin(
        {"no_score_rule", "ambiguous", "unknown"}
    )
    blend = pd.Series(np.where(low_info_mask, 0.60, 0.20), index=hidden.index, dtype=float)
    anchors = anchors.mul(1.0 - blend, axis=0) + stage.mul(blend, axis=0)
    anchors = anchors + _EPS
    anchors = anchors.div(anchors.sum(axis=1).replace(0.0, 1.0), axis=0)

    hidden["anchor_s1"] = anchors["anchor_s1"]
    hidden["anchor_s2"] = anchors["anchor_s2"]
    hidden["anchor_s3"] = anchors["anchor_s3"]

    hidden["observation_resolution_type"] = _safe_text_series(frame, "observation_resolution_type", "unknown")
    hidden["observation_confidence_label"] = _safe_text_series(frame, "observation_confidence_label", "low")
    hidden["observation_quality_flag"] = _safe_text_series(frame, "observation_quality_flag", "unknown")
    hidden["sequence_quality_flag"] = _safe_text_series(frame, "sequence_quality_flag", "low")
    hidden["sequence_resolution_type"] = _safe_text_series(frame, "sequence_resolution_type", "fallback")
    hidden["is_train_eligible"] = (
        pd.to_numeric(
            frame.get("is_train_eligible", pd.Series([0.0] * len(frame), index=frame.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )
    hidden["train_weight"] = _build_train_weight(frame)

    observed = (
        frame.get("observed_zap_class", pd.Series(["unknown"] * len(frame), index=frame.index))
        .fillna("unknown")
        .astype(str)
    )

    for col in HIDDEN_STATE_FEATURE_COLUMNS:
        if col not in hidden.columns:
            hidden[col] = 0.0

    return HiddenStateFeatureLayer(hidden_state_features=hidden, observed_sequence=observed)
