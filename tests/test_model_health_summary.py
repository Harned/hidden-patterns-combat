from __future__ import annotations

from pathlib import Path

import pandas as pd

from hidden_patterns_combat.diagnostics.model_health import build_model_health_summary, write_model_health_summary


def test_model_health_summary_marks_degenerate_and_partial_semantics(tmp_path: Path) -> None:
    analysis = pd.DataFrame(
        {
            "hidden_state": [0] * 95 + [1] * 5,
            "observed_zap_class": ["no_score"] * 100,
        }
    )
    transitions = [
        {"from_state": 0, "to_state": 0, "count": 99, "share": 0.99, "is_self_loop": True},
        {"from_state": 0, "to_state": 1, "count": 1, "share": 0.01, "is_self_loop": False},
    ]
    canonical_map = {
        "n_states": 3,
        "semantic_assignment": {"S1": 0},
        "semantic_confidence": {"S1": 0.6, "S2": 0.0, "S3": 0.0},
    }
    observed_summary = {
        "direct_share": 0.0,
        "no_score_rule_share": 0.90,
        "unknown_share": 0.05,
        "ambiguous_share": 0.05,
    }
    state_profile = pd.DataFrame(
        {
            "hidden_state": [0, 1, 2],
            "key_link": ["maneuvering", "maneuvering", "maneuvering"],
        }
    )

    result = build_model_health_summary(
        analysis_df=analysis,
        transitions=transitions,
        canonical_map=canonical_map,
        observed_summary=observed_summary,
        state_profile=state_profile,
    )

    summary = result.summary
    assert float(summary["self_transition_share"]) >= 0.95
    assert summary["degenerate_transition_warning"] is True
    assert summary["low_information_observed_layer_warning"] is True
    assert summary["semantic_assignment_quality"] == "partial"
    assert summary["semantic_assignment_quality_legacy"] == "partial_semantic_assignment"
    assert "S1" in summary["semantic_confirmed_states"]
    assert summary["maneuvering_only_state_profile_warning"] is True

    written = write_model_health_summary(result, diagnostics_dir=tmp_path)
    assert Path(written).exists()


def test_model_health_summary_marks_failed_when_assignment_has_no_confirmed_states() -> None:
    analysis = pd.DataFrame(
        {
            "hidden_state": [0, 1, 2, 0, 1, 2],
            "observed_zap_class": ["no_score"] * 6,
        }
    )
    transitions = [
        {"from_state": 0, "to_state": 1, "count": 2, "share": 0.4, "is_self_loop": False},
        {"from_state": 1, "to_state": 2, "count": 2, "share": 0.4, "is_self_loop": False},
        {"from_state": 2, "to_state": 2, "count": 1, "share": 0.2, "is_self_loop": True},
    ]
    canonical_map = {
        "n_states": 3,
        "semantic_assignment": {"S1": 0, "S2": 1, "S3": 2},
        "semantic_confidence": {"S1": 0.1, "S2": 0.2, "S3": 0.3},
    }
    observed_summary = {
        "direct_share": 0.0,
        "no_score_rule_share": 0.5,
        "unknown_share": 0.3,
        "ambiguous_share": 0.2,
    }
    state_profile = pd.DataFrame({"hidden_state": [0, 1, 2], "key_link": ["maneuvering", "kfv", "vup"]})

    result = build_model_health_summary(
        analysis_df=analysis,
        transitions=transitions,
        canonical_map=canonical_map,
        observed_summary=observed_summary,
        state_profile=state_profile,
    )
    summary = result.summary
    assert summary["semantic_assignment_quality"] == "failed"
    assert summary["semantic_confirmed_states"] == []
    assert int(summary["semantic_unconfirmed_states_count"]) == 3
