import pandas as pd

from hidden_patterns_combat.config import FeatureConfig
from hidden_patterns_combat.features.encoder import encode_features


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
    assert batch.features["maneuver_right_code"].tolist() == [1, 2]
    assert batch.features["kfv_code"].tolist() == [1, 1]
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
    assert batch.features["maneuver_right_code"].tolist() == [1, 2]
    assert batch.features["maneuver_left_code"].tolist() == [1, 1]
