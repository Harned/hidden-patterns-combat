from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ModelHealthResult:
    summary: dict[str, Any]


def _semantic_assignment_quality(semantic_assignment: dict[str, int]) -> str:
    required = {"S1", "S2", "S3"}
    assigned = {str(k) for k, v in semantic_assignment.items() if v is not None}
    assigned_required = required.intersection(assigned)
    if len(assigned_required) >= 3:
        return "full_semantic_assignment"
    if len(assigned_required) >= 1:
        return "partial_semantic_assignment"
    return "failed_semantic_assignment"


def build_model_health_summary(
    analysis_df: pd.DataFrame,
    *,
    transitions: list[dict[str, object]],
    canonical_map: dict[str, object],
    observed_summary: dict[str, float],
    state_profile: pd.DataFrame,
) -> ModelHealthResult:
    frame = analysis_df.copy().reset_index(drop=True)

    transition_total = int(sum(int(row.get("count", 0)) for row in transitions))
    self_transition_total = int(
        sum(
            int(row.get("count", 0))
            for row in transitions
            if bool(row.get("is_self_loop", False))
        )
    )
    self_transition_share = float(self_transition_total / transition_total) if transition_total > 0 else 0.0
    self_loop_rows = [row for row in transitions if bool(row.get("is_self_loop", False))]
    top_self_transition_share = float(max((float(row.get("share", 0.0)) for row in self_loop_rows), default=0.0))

    n_states = int(canonical_map.get("n_states", 0) or 0)
    if n_states <= 0 and "hidden_state" in frame.columns:
        n_states = int(frame["hidden_state"].nunique(dropna=True))
    n_states = max(1, n_states)

    state_usage_share = {}
    effective_state_usage = 0.0
    if "hidden_state" in frame.columns and not frame.empty:
        counts = frame["hidden_state"].astype(int).value_counts(normalize=True).to_dict()
        state_usage_share = {int(k): float(v) for k, v in counts.items()}
        effective_count = sum(1 for v in counts.values() if float(v) >= 0.05)
        effective_state_usage = float(effective_count / n_states)

    semantic_assignment = {
        str(k): int(v) for k, v in (canonical_map.get("semantic_assignment", {}) or {}).items()
    }
    semantic_confidence = {
        str(k): float(v) for k, v in (canonical_map.get("semantic_confidence", {}) or {}).items()
    }
    semantic_quality = _semantic_assignment_quality(semantic_assignment)

    low_information_observed_layer_warning = bool(
        observed_summary.get("direct_share", 0.0) < 0.05
        and observed_summary.get("no_score_rule_share", 0.0) > 0.70
    )
    degenerate_transition_warning = bool(self_transition_share >= 0.95 or effective_state_usage <= 0.34)

    state_profile_links = (
        state_profile.get("key_link", pd.Series([], dtype="object")).fillna("").astype(str).str.strip().tolist()
    )
    unique_links = sorted({x for x in state_profile_links if x})
    maneuvering_only_profile_warning = bool(unique_links and set(unique_links) == {"maneuvering"})

    warnings: list[str] = []
    if low_information_observed_layer_warning:
        warnings.append(
            "Observed layer is low-information: direct finish share is very low and no_score dominates."
        )
    if degenerate_transition_warning:
        warnings.append(
            "Hidden-state transitions are close to degenerate (very high self-loop share or low effective state usage)."
        )
    if maneuvering_only_profile_warning:
        warnings.append("State profiles collapse to maneuvering-like links; semantic contrast is weak.")
    if semantic_quality != "full_semantic_assignment":
        warnings.append(
            "State semantics did not stabilize sufficiently for confident KFV/VUP interpretation."
        )

    summary: dict[str, Any] = {
        "rows_total": int(len(frame)),
        "n_states": int(n_states),
        "self_transition_share": self_transition_share,
        "top_self_transition_share": top_self_transition_share,
        "effective_state_usage": effective_state_usage,
        "state_usage_share": {str(k): float(v) for k, v in state_usage_share.items()},
        "semantic_assignment": semantic_assignment,
        "semantic_confidence": semantic_confidence,
        "semantic_assignment_quality": semantic_quality,
        "degenerate_transition_warning": degenerate_transition_warning,
        "low_information_observed_layer_warning": low_information_observed_layer_warning,
        "maneuvering_only_state_profile_warning": maneuvering_only_profile_warning,
        "warnings": warnings,
    }
    return ModelHealthResult(summary=summary)


def write_model_health_summary(
    result: ModelHealthResult,
    diagnostics_dir: str | Path,
) -> str:
    out = Path(diagnostics_dir)
    out.mkdir(parents=True, exist_ok=True)

    path = out / "model_health_summary.json"
    path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)
