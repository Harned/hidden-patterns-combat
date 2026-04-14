import pandas as pd
import pytest

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


def test_read_excel_sheets_auto_detects_matrix_parser(matrix_excel_path):
    sheets = read_excel_sheets(matrix_excel_path, sheets=["Matrix"])
    assert sheets[0].parser_type == "matrix"
    assert len(sheets[0].dataframe) == 3


def test_read_excel_sheets_auto_falls_back_to_table(demo_excel_path):
    sheets = read_excel_sheets(demo_excel_path, sheets=["Общее"], parser_mode="auto")
    assert sheets[0].parser_type == "table"


def test_read_excel_sheets_matrix_mode_raises_for_non_matrix_sheet(demo_excel_path):
    with pytest.raises(ValueError, match="does not match matrix-style markers"):
        read_excel_sheets(demo_excel_path, sheets=["Общее"], parser_mode="matrix")
