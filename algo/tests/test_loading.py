from __future__ import annotations

from pathlib import Path

import pytest

from hpc_algo.loading import ExcelLoadError, load_excel


def test_load_excel_reads_sheets(zap_only_excel: Path) -> None:
    loaded = load_excel(zap_only_excel)
    assert loaded.sheet_names == ["Общее"]
    assert "Общее" in loaded.sheets
    assert loaded.size_bytes > 0
    assert len(loaded.sha256) == 64


def test_load_excel_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ExcelLoadError):
        load_excel(tmp_path / "does_not_exist.xlsx")


def test_load_excel_wrong_extension(tmp_path: Path) -> None:
    wrong = tmp_path / "not_excel.txt"
    wrong.write_text("hello")
    with pytest.raises(ExcelLoadError):
        load_excel(wrong)
