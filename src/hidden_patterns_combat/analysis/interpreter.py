from __future__ import annotations

import pandas as pd


def state_profile_table(features: pd.DataFrame, states: pd.Series) -> pd.DataFrame:
    merged = features.copy()
    merged["hidden_state"] = states.values
    grouped = merged.groupby("hidden_state", dropna=False).mean(numeric_only=True)
    grouped["episodes_count"] = merged.groupby("hidden_state").size()
    return grouped.reset_index()


def text_summary(profile: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in profile.iterrows():
        lines.append(
            "State {s}: n={n}, maneuverR={mr:.2f}, maneuverL={ml:.2f}, "
            "KFV={kfv:.2f}, VUP={vup:.2f}, result={res:.2f}".format(
                s=int(row["hidden_state"]),
                n=int(row.get("episodes_count", 0)),
                mr=row.get("maneuver_right_code", 0.0),
                ml=row.get("maneuver_left_code", 0.0),
                kfv=row.get("kfv_code", 0.0),
                vup=row.get("vup_code", 0.0),
                res=row.get("observed_result", 0.0),
            )
        )
    return "\n".join(lines)
