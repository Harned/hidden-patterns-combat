from __future__ import annotations

import pandas as pd

from .state_definition import StateDefinition


def _dominant_block_label(row: pd.Series) -> str:
    candidates = {
        "maneuvering": float(row.get("maneuver_right_code", 0.0) + row.get("maneuver_left_code", 0.0)),
        "kfv": float(row.get("kfv_code", 0.0)),
        "vup": float(row.get("vup_code", 0.0)),
        "outcome_actions": float(row.get("outcome_actions_code", 0.0)),
    }
    return max(candidates, key=candidates.get)


def _posthoc_text(row: pd.Series) -> str:
    dominant = _dominant_block_label(row)
    if dominant == "kfv":
        return "Пост-хок: состояние характеризуется повышенной активностью КФВ."
    if dominant == "vup":
        return "Пост-хок: состояние характеризуется относительно выраженным ВУП компонентом."
    if dominant == "maneuvering":
        return "Пост-хок: состояние смещено в сторону стойки/маневрирования."
    if dominant == "outcome_actions":
        return "Пост-хок: состояние связано с завершающими действиями."
    return "Пост-хок интерпретация не определена."


def interpret_decoded_states(
    engineered_features: pd.DataFrame,
    decoded_states: pd.Series,
    state_definition: StateDefinition,
) -> pd.DataFrame:
    frame = engineered_features.copy()
    frame["state_id"] = decoded_states.values
    out = frame.groupby("state_id", dropna=False).mean(numeric_only=True)
    out["episodes_count"] = frame.groupby("state_id").size()
    out = out.reset_index()
    out["state_name"] = out["state_id"].apply(state_definition.state_name)
    out["raw_hidden_state"] = out["state_name"]
    out["posthoc_interpretation"] = out.apply(_posthoc_text, axis=1)
    return out
