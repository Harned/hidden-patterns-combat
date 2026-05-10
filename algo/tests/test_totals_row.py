"""Регрессии для фильтра строк-итогов в audit (Phase 5).

Реальные файлы вида ``docs/Оценка СД содержание.xlsx`` иногда содержат
хвостовую строку «Итого» с суммами по числовым колонкам. Структурно
такая строка идентична обычной строке эпизода: те же колонки
заполнены, но это сумма, а не наблюдение. Без фильтра она попадает
в baseline/HMM и искажает распределения и переходы.

Тесты проверяют:

* :func:`hpc_algo.audit.detect_totals_row_indices` находит строку
  «Итого» по строковому маркеру в любой колонке;
* :func:`hpc_algo.audit.build_audit` отмечает позицию итоговой строки
  в ``SheetAudit.totals_row_indices`` и добавляет суспект-сообщение;
* :func:`hpc_algo.api.analyze_source` исключает итоги из mapped
  frames до baseline/HMM и эмитит warning ``audit.totals_row_detected``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.audit import (
    build_audit,
    detect_totals_row_indices,
    drop_totals_rows,
)
from hpc_algo.loading import load_excel


def test_detect_totals_row_indices_marks_known_marker() -> None:
    df = pd.DataFrame(
        {
            "ФИО борца": ["Иванов", "Петров", "Итого"],
            "n": [1, 2, 3],
            "score": [4, 5, 9],
        }
    )
    assert detect_totals_row_indices(df) == [2]


def test_detect_totals_row_indices_ignores_substring_match() -> None:
    """«Итогов» как фамилия не должен срабатывать как маркер итогов."""

    df = pd.DataFrame(
        {
            "ФИО борца": ["Иванов", "Итогов", "Петров"],
            "n": [1, 2, 3],
        }
    )
    assert detect_totals_row_indices(df) == []


def test_drop_totals_rows_returns_clean_copy() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = drop_totals_rows(df, [1])
    assert list(out["a"]) == [1, 3]
    # Исходный df не модифицирован.
    assert list(df["a"]) == [1, 2, 3]


def test_build_audit_records_totals_indices(totals_row_excel: Path) -> None:
    loaded = load_excel(totals_row_excel)
    audit = build_audit(loaded)
    sheet = audit.sheets[0]
    assert sheet.totals_row_indices, sheet.totals_row_indices
    assert any("итог" in s.lower() for s in sheet.suspicious)


def test_analyze_source_drops_totals_row_with_warning(
    totals_row_excel: Path,
) -> None:
    cfg = preflight_mapping(totals_row_excel)
    result = analyze_source(totals_row_excel, AnalyzeConfig(column_mapping=cfg))
    codes = {w.code for w in result.warnings}
    assert "audit.totals_row_detected" in codes
    info = next(
        w.context for w in result.warnings
        if w.code == "audit.totals_row_detected"
    )
    assert info["n_rows_dropped"] == 1
    assert info["rows_after"] == info["rows_before"] - 1
