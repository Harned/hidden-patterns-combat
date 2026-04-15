from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
)


@dataclass
class HiddenStateFeatureLayer:
    hidden_state_features: pd.DataFrame
    observed_sequence: pd.Series


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def build_hidden_state_feature_layer(canonical_episode_table: pd.DataFrame) -> HiddenStateFeatureLayer:
    frame = canonical_episode_table.copy().reset_index(drop=True)

    hidden = pd.DataFrame(index=frame.index)
    for col in HIDDEN_STATE_FEATURE_COLUMNS:
        if col in frame.columns:
            hidden[col] = _safe_numeric_series(frame[col])
        else:
            hidden[col] = 0.0

    observed = (
        frame.get("observed_zap_class", pd.Series(["unknown"] * len(frame), index=frame.index))
        .fillna("unknown")
        .astype(str)
    )

    return HiddenStateFeatureLayer(hidden_state_features=hidden, observed_sequence=observed)
