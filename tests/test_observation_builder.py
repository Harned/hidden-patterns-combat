from __future__ import annotations

import pandas as pd

from hidden_patterns_combat.preprocessing.observation_builder import build_observed_zap_classes


def test_observation_mapping_direct_finish_high_confidence() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [2],
            "outcomes__удержание": [1],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "hold"
    assert out.loc[0, "observation_resolution_type"] == "direct_finish_signal"
    assert out.loc[0, "observation_confidence_label"] == "high"


def test_observation_mapping_score_fallback_medium_confidence() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [1],
            "outcomes__удержание": [0],
            "outcomes__болевой_на_руку": [0],
            "outcomes__болевой_на_ногу": [0],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "zap_r"
    assert out.loc[0, "observation_resolution_type"] == "inferred_from_score"
    assert out.loc[0, "observation_confidence_label"] == "medium"
    assert out.loc[0, "observation_quality_flag"] == "ok_score_rule"


def test_observation_mapping_score_zero_without_finish_is_no_score_high() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [0],
            "outcomes__finish_action_hold": [0],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "no_score"
    assert out.loc[0, "observation_resolution_type"] == "no_score_rule"
    assert out.loc[0, "observation_confidence_label"] == "high"


def test_observation_mapping_ambiguous_finish_low_confidence() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [2],
            "outcomes__удержание": [1],
            "outcomes__болевой_на_руку": [1],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "unknown"
    assert out.loc[0, "observation_resolution_type"] == "ambiguous"
    assert out.loc[0, "observation_confidence_label"] == "low"
    assert out.loc[0, "observation_quality_flag"] == "unknown_ambiguous_finish"


def test_observation_mapping_unknown_when_missing_inputs() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [None],
            "outcomes__удержание": [0],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "unknown"
    assert out.loc[0, "observation_resolution_type"] == "unknown"
    assert out.loc[0, "observation_confidence_label"] == "low"


def test_observation_mapping_supports_positional_finish_action_columns() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [0, 0, 0],
            "outcomes__finish_action_04_05": [1, 0, 0],
            "outcomes__finish_action_05_06": [0, 1, 0],
            "outcomes__finish_action_06_07": [0, 0, 1],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out["observed_zap_class"].tolist() == ["hold", "arm_submission", "leg_submission"]
    assert (out["observation_resolution_type"] == "direct_finish_signal").all()
