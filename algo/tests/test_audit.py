from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.audit import build_audit, filter_audit_to_sheets
from hpc_algo.loading import load_excel


def test_audit_counts_rows_and_columns(zap_only_excel: Path) -> None:
    loaded = load_excel(zap_only_excel)
    audit = build_audit(loaded)

    assert len(audit.sheets) == 1
    sheet = audit.sheets[0]
    assert sheet.name == "Общее"
    assert sheet.n_rows == 4
    assert sheet.n_cols == 4
    assert {c.name for c in sheet.columns} == {
        "Спортсмен",
        "Эпизод",
        "Время, сек",
        "ЗАП",
    }


def test_audit_preview_non_empty(zap_only_excel: Path) -> None:
    loaded = load_excel(zap_only_excel)
    audit = build_audit(loaded)
    preview = audit.sheets[0].preview
    assert preview, "preview должен содержать несколько первых строк"
    assert "ЗАП" in preview[0]


def test_audit_reports_empty_sheet(empty_excel: Path) -> None:
    loaded = load_excel(empty_excel)
    audit = build_audit(loaded)
    assert audit.total_rows == 0
    assert any("пуст" in s.lower() for s in audit.sheets[0].suspicious)


def test_audit_overall_null_ratio_bounded(full_structure_excel: Path) -> None:
    loaded = load_excel(full_structure_excel)
    audit = build_audit(loaded)
    assert 0.0 <= audit.overall_null_ratio <= 1.0


def test_filter_audit_to_sheets_keeps_subset(tmp_path: Path) -> None:
    path = tmp_path / "two_sheets.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_excel(
            writer, sheet_name="48", index=False
        )
        pd.DataFrame({"a": [1, 2], "b": [3, None]}).to_excel(
            writer, sheet_name="52", index=False
        )

    loaded = load_excel(path)
    audit = build_audit(loaded)
    assert {s.name for s in audit.sheets} == {"48", "52"}

    scoped = filter_audit_to_sheets(audit, {"48"})

    assert [s.name for s in scoped.sheets] == ["48"]
    sheet48 = next(s for s in audit.sheets if s.name == "48")
    assert scoped.total_rows == sheet48.n_rows
    assert scoped.total_cells == sheet48.n_rows * sheet48.n_cols
    assert 0.0 <= scoped.overall_null_ratio <= 1.0


def test_filter_audit_to_sheets_empty_set_returns_zero(zap_only_excel: Path) -> None:
    loaded = load_excel(zap_only_excel)
    audit = build_audit(loaded)
    scoped = filter_audit_to_sheets(audit, set())
    assert scoped.sheets == []
    assert scoped.total_rows == 0
    assert scoped.total_cells == 0
    assert scoped.overall_null_ratio == 0.0
