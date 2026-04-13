import pandas as pd

from hidden_patterns_combat.config import FeatureConfig
from hidden_patterns_combat.features.engineering import FeatureEngineer, FeatureEngineeringConfig


def test_feature_engineering_produces_kfv_subgroups_and_traceability():
    cols = {
        "metadata__episode_attr_01": [1, 2],
        "outcomes__score": [2, 0],
    }
    for i in range(1, 25):
        cols[f"maneuvering__indicator_{i:02d}"] = [1 if i % 2 else 0, 0 if i % 2 else 1]
    for i in range(1, 30):
        cols[f"kfv__indicator_{i:02d}"] = [1 if i in (1, 7, 13, 19, 25) else 0, 0]
    for i in range(1, 6):
        cols[f"vup__indicator_{i:02d}"] = [1 if i == 1 else 0, 0]
    for i in range(1, 7):
        cols[f"outcomes__finish_action_{i:02d}"] = [1 if i == 2 else 0, 0]

    df = pd.DataFrame(cols)
    res = FeatureEngineer(FeatureConfig(), FeatureEngineeringConfig()).transform(df)

    engineered = res.engineered_feature_set
    assert "grips_code" in engineered.columns
    assert "holds_code" in engineered.columns
    assert "bodylocks_code" in engineered.columns
    assert "underhooks_code" in engineered.columns
    assert "posts_code" in engineered.columns
    assert "kfv_code" in engineered.columns
    assert "outcome_actions_code" in engineered.columns

    trace = res.traceability
    assert (trace["engineered_feature"] == "grips_code").any()
    assert res.validation.is_valid is True


def test_feature_engineering_validation_marks_missing_groups():
    df = pd.DataFrame({"metadata__athlete_name": ["A", "B"]})
    res = FeatureEngineer(FeatureConfig()).transform(df)
    assert res.validation.is_valid is False
    assert "kfv" in res.validation.missing_groups
    assert "vup" in res.validation.missing_groups


def test_feature_engineering_handles_incorrect_types(cleaned_like_df):
    res = FeatureEngineer(FeatureConfig()).transform(cleaned_like_df)
    engineered = res.engineered_feature_set
    assert engineered.shape[0] == len(cleaned_like_df)
    assert engineered["maneuver_right_code"].dtype.kind in ("i", "u", "f")
    assert engineered["observed_result"].fillna(0).ge(0).all()
    assert "athlete_name" in res.metadata.columns
    assert "sequence_id" in res.metadata.columns
