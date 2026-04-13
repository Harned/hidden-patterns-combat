import pandas as pd
from sklearn.preprocessing import StandardScaler

from hidden_patterns_combat.modeling.observation_encoding import build_lengths, encode_observations


def test_build_lengths_from_sequence_ids():
    s = pd.Series(["a", "a", "b", "b", "b", "c"])
    assert build_lengths(s) == [2, 3, 1]


def test_encode_observations_keeps_feature_order_and_lengths():
    df = pd.DataFrame({"f1": [1, 2, 3], "f2": [0, 1, 0]})
    seq = pd.Series(["x", "x", "y"])
    batch = encode_observations(df, scaler=StandardScaler(), fit_scaler=True, sequence_ids=seq)
    assert batch.feature_columns == ["f1", "f2"]
    assert batch.values.shape == (3, 2)
    assert batch.lengths == [2, 1]


def test_build_lengths_empty_series():
    s = pd.Series([], dtype=object)
    assert build_lengths(s) == []
