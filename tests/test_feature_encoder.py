import pandas as pd
import pytest

from hidden_patterns_combat.config import FeatureConfig
from hidden_patterns_combat.features.encoder import encode_features, select_hmm_input_features


def test_encode_features_compact_code_and_metadata():
    df = pd.DataFrame(
        {
            "номер эпизода": [1, 2],
            "стойка прав | x1": [1, 0],
            "стойка прав | x2": [0, 1],
            "стойка лев | y1": [0, 1],
            "кфв | захват": [1, 1],
            "вуп | z1": [0, 1],
            "время эпизода": [30, 45],
            "баллы": [2, 0],
        }
    )
    batch = encode_features(df, FeatureConfig())
    assert list(batch.metadata["episode_id"]) == ["1", "2"]
    assert batch.features["maneuver_right_code"].iloc[0] == pytest.approx(1.0)
    assert batch.features["maneuver_right_code"].iloc[1] == pytest.approx(1.5849625, rel=1e-6)
    assert batch.features["kfv_code"].tolist() == [1.0, 1.0]
    assert batch.features["observed_result"].tolist() == [2.0, 0.0]


def test_encode_features_fallback_splits_maneuver_block():
    df = pd.DataFrame(
        {
            "стойка и маневрирование самбиста (основные в эпизоде)": [1, 0],
            "стойка и маневрирование самбиста (основные в эпизоде)_2": [0, 1],
            "стойка и маневрирование самбиста (основные в эпизоде)_3": [1, 1],
            "стойка и маневрирование самбиста (основные в эпизоде)_4": [0, 0],
        }
    )
    batch = encode_features(df, FeatureConfig())
    assert batch.features["maneuver_right_code"].iloc[0] == pytest.approx(1.0)
    assert batch.features["maneuver_right_code"].iloc[1] == pytest.approx(1.5849625, rel=1e-6)
    assert batch.features["maneuver_left_code"].tolist() == [1.0, 1.0]


def test_encode_features_with_missing_expected_columns():
    df = pd.DataFrame({"metadata__athlete_name": ["A", "B"]})
    batch = encode_features(df, FeatureConfig())
    assert (batch.features["kfv_code"] == 0).all()
    assert (batch.features["vup_code"] == 0).all()
    assert list(batch.metadata["episode_id"]) == ["0", "1"]


def test_select_hmm_input_features_excludes_outcome_columns():
    features = pd.DataFrame(
        {
            "maneuver_right_code": [1.0],
            "kfv_code": [2.0],
            "vup_code": [0.5],
            "outcome_actions_code": [1.0],
            "observed_result": [3.0],
        }
    )
    hmm_x = select_hmm_input_features(features)
    assert "outcome_actions_code" not in hmm_x.columns
    assert "observed_result" not in hmm_x.columns
    assert set(hmm_x.columns) == {"maneuver_right_code", "kfv_code", "vup_code"}
