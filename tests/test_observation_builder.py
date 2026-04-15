from __future__ import annotations

import pandas as pd

from hidden_patterns_combat.preprocessing.observation_builder import build_observed_zap_classes


def test_observation_mapping_score_zero_without_finish_is_no_score() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [0],
            "outcomes__finish_action_hold": [0],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "no_score"
    assert out.loc[0, "observation_quality_flag"] == "ok_no_score_rule"


def test_observation_mapping_hold_column_is_hold() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [2],
            "outcomes__удержание": [1],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "hold"
    assert str(out.loc[0, "observation_quality_flag"]).startswith("ok_finish_rule")


def test_observation_mapping_arm_submission() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [4],
            "outcomes__болевой_на_руку": [1],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "arm_submission"


def test_observation_mapping_leg_submission() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [4],
            "outcomes__болевой_на_ногу": [1],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "leg_submission"


def test_observation_mapping_ambiguous_row_to_unknown_with_quality_flag() -> None:
    df = pd.DataFrame(
        {
            "outcomes__score": [2],
            "outcomes__удержание": [1],
            "outcomes__болевой_на_руку": [1],
        }
    )
    out = build_observed_zap_classes(df).observations
    assert out.loc[0, "observed_zap_class"] == "unknown"
    assert out.loc[0, "observation_quality_flag"] == "unknown_ambiguous_finish"
