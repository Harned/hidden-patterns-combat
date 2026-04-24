from __future__ import annotations

from pathlib import Path

from hpc_algo.audit import build_audit
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
