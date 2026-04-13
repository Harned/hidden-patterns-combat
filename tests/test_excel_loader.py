import pandas as pd

from hidden_patterns_combat.io.excel_loader import flatten_columns, read_excel_sheets


def test_flatten_columns_multiindex_deduplicates():
    columns = pd.MultiIndex.from_tuples(
        [("Маневрирование", "Стойка прав", "A"), ("Маневрирование", "Стойка прав", "A")]
    )
    flat = flatten_columns(columns)
    assert flat[0] == "маневрирование | стойка прав | a"
    assert flat[1].startswith("маневрирование | стойка прав | a_")


def test_flatten_columns_handles_unnamed_levels():
    columns = pd.MultiIndex.from_tuples([("Unnamed: 0_level_0", "Баллы"), ("", "")])
    flat = flatten_columns(columns)
    assert flat[0] == "баллы"
    assert flat[1] == "unknown_column"


def test_read_excel_sheets_multirow_headers(demo_excel_path):
    sheets = read_excel_sheets(demo_excel_path, sheets=["Общее"], header_depth=2)
    df = sheets[0].dataframe
    # Check duplicate header was de-duplicated and rows were read.
    assert any(col.startswith("стойка и маневрирование самбиста") for col in df.columns)
    assert len(df) == 2
