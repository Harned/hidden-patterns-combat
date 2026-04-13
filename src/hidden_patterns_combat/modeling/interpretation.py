from __future__ import annotations

import pandas as pd

from .state_definition import StateDefinition


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
    return out
