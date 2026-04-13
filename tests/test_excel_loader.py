import pandas as pd

from hidden_patterns_combat.io.excel_loader import flatten_columns


def test_flatten_columns_multiindex_deduplicates():
    columns = pd.MultiIndex.from_tuples(
        [("Маневрирование", "Стойка прав", "A"), ("Маневрирование", "Стойка прав", "A")]
    )
    flat = flatten_columns(columns)
    assert flat[0] == "маневрирование | стойка прав | a"
    assert flat[1].startswith("маневрирование | стойка прав | a_")
