from __future__ import annotations

import pandas as pd

from hidden_patterns_combat.modeling.state_definition import StateDefinition


def state_profile_table(
    features: pd.DataFrame,
    states: pd.Series,
    state_definition: StateDefinition | None = None,
) -> pd.DataFrame:
    merged = features.copy()
    merged["hidden_state"] = states.values
    grouped = merged.groupby("hidden_state", dropna=False).mean(numeric_only=True)
    grouped["episodes_count"] = merged.groupby("hidden_state").size()
    out = grouped.reset_index()
    if state_definition is not None:
        out["hidden_state_name"] = out["hidden_state"].astype(int).apply(state_definition.state_name)
    return out


def text_summary(profile: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in profile.iterrows():
        state_name = str(row.get("hidden_state_name", f"state_{int(row['hidden_state'])}"))
        lines.append(
            "Latent state {name}: n={n}, maneuverR={mr:.2f}, maneuverL={ml:.2f}, "
            "KFV={kfv:.2f}, VUP={vup:.2f}, result={res:.2f}".format(
                name=state_name,
                n=int(row.get("episodes_count", 0)),
                mr=row.get("maneuver_right_code", 0.0),
                ml=row.get("maneuver_left_code", 0.0),
                kfv=row.get("kfv_code", 0.0),
                vup=row.get("vup_code", 0.0),
                res=row.get("observed_result", 0.0),
            )
        )
    return "\n".join(lines)
