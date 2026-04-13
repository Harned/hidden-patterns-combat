import pandas as pd

from hidden_patterns_combat.preprocessing.headers import (
    deduplicate_columns,
    normalize_column_token,
    normalize_columns,
)


def test_normalize_columns_dedupes_and_lowercases():
    df = pd.DataFrame([[1, 2]], columns=[" Баллы ", "баллы"])
    out = normalize_columns(df)
    assert out.columns.tolist() == ["баллы", "баллы_2"]


def test_deduplicate_columns_unknown_fallback():
    cols = deduplicate_columns(["", "", "x", "x"])
    assert cols == ["unknown_column", "unknown_column_2", "x", "x_2"]


def test_normalize_column_token_collapses_spaces_and_pipes():
    token = normalize_column_token("  A   |   B   || C  ")
    assert token == "a | b | c"
