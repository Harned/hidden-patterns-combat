from __future__ import annotations

import pandas as pd

from hidden_patterns_combat.io.excel_loader import read_excel_sheets
from hidden_patterns_combat.io.matrix_parser import (
    detect_matrix_episode_sheet,
    load_matrix_episode_sheet,
    normalize_matrix_feature_label,
)
from hidden_patterns_combat.preprocessing.ingestion import load_excel_for_preprocessing


def test_matrix_sheet_parses_to_tidy_rows(matrix_excel_path):
    sheets = read_excel_sheets(matrix_excel_path, sheets=["Matrix"])
    assert len(sheets) == 1
    assert sheets[0].parser_type == "matrix"

    tidy = sheets[0].dataframe
    assert len(tidy) == 3
    assert "episode_id" in tidy.columns
    assert "episode_duration" in tidy.columns
    assert "pause_duration" in tidy.columns
    assert "observed_result" in tidy.columns
    assert any(c.startswith("maneuver_right_indicator_") for c in tidy.columns)
    assert any(c.startswith("kfv_grips_indicator_") for c in tidy.columns)


def test_detection_distinguishes_matrix_and_tabular(matrix_excel_path, demo_excel_path):
    matrix_raw = pd.read_excel(matrix_excel_path, sheet_name="Matrix", header=None, engine="openpyxl")
    tabular_raw = pd.read_excel(demo_excel_path, sheet_name="Общее", header=None, engine="openpyxl")

    assert detect_matrix_episode_sheet(matrix_raw) is True
    assert detect_matrix_episode_sheet(tabular_raw) is False


def test_label_normalization_mvp_rules():
    assert normalize_matrix_feature_label("№ эпизода") == "episode_id"
    assert normalize_matrix_feature_label("Время эпизода") == "episode_duration"
    assert normalize_matrix_feature_label("Баллы") == "observed_result"


def test_force_matrix_parser_option(matrix_excel_path):
    ingestion = load_excel_for_preprocessing(
        matrix_excel_path,
        sheet_selector="Matrix",
        force_matrix_parser=True,
    )
    assert ingestion.episodes_tidy is not None
    assert len(ingestion.episodes_tidy) == 3
    assert ingestion.matrix_label_mapping is not None


def test_load_matrix_episode_sheet_returns_mapping(matrix_excel_path):
    raw = pd.read_excel(matrix_excel_path, sheet_name="Matrix", header=None, engine="openpyxl")
    parsed = load_matrix_episode_sheet(raw, sheet_name="Matrix")
    assert not parsed.tidy.empty
    assert not parsed.label_mapping.empty
    assert set(["source_label", "normalized_column"]).issubset(set(parsed.label_mapping.columns))
