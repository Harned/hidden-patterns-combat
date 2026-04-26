"""Тесты эвристики forward-fill по колонке ФИО."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from hpc_algo.suggestions import (
    forward_fill_athlete_suggestions,
    header_merge_fill_suggestions,
)


@pytest.fixture
def athlete_xlsx(tmp_path: Path) -> Path:
    """Лист с одной строкой заголовка и тремя data-строками: ФИО только в первой."""

    path = tmp_path / "athlete.xlsx"
    df = pd.DataFrame(
        [
            ["Иванов", 1, 10],
            [None, 2, 20],
            ["", 3, 30],
            ["Петров", 1, 5],
            [None, 2, 7],
        ],
        columns=["ФИО борца", "№ эпизода", "Время эпизода, с."],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    return path


def test_forward_fill_proposes_each_blank_cell(athlete_xlsx: Path) -> None:
    report = forward_fill_athlete_suggestions(
        athlete_xlsx,
        "Sheet1",
        header_rows=[0],
        athlete_columns=["ФИО борца"],
    )
    assert report.athlete_column == "ФИО борца"
    assert report.warning is None
    rows = [(s.row, s.proposed, s.source_row) for s in report.suggestions]
    # header_rows=[0] → first data Excel row = 2.
    # Строка 3 (Excel) пустая → предложить «Иванов» из строки 2.
    # Строка 4 (Excel) пустая (пустая строка) → предложить «Иванов» из строки 2.
    # Строка 5 — «Петров», обновляет источник.
    # Строка 6 пустая → предложить «Петров» из строки 5.
    assert rows == [
        (3, "Иванов", 2),
        (4, "Иванов", 2),
        (6, "Петров", 5),
    ]
    # Все предложения относятся к одной и той же 1-based колонке Excel.
    assert {s.col for s in report.suggestions} == {1}


def test_forward_fill_skips_until_first_value(tmp_path: Path) -> None:
    path = tmp_path / "no_first.xlsx"
    df = pd.DataFrame(
        [
            [None, 1],
            [None, 2],
            ["Сидоров", 3],
            [None, 4],
        ],
        columns=["ФИО борца", "№ эпизода"],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)

    report = forward_fill_athlete_suggestions(
        path,
        "Sheet1",
        header_rows=[0],
        athlete_columns=["ФИО борца"],
    )
    rows = [(s.row, s.proposed) for s in report.suggestions]
    assert rows == [(5, "Сидоров")]


def test_forward_fill_handles_multirow_header(tmp_path: Path) -> None:
    path = tmp_path / "multirow.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Спортсмен", "Эпизод"])
    ws.append(["ФИО", "№"])
    ws.append(["Иванов", 1])
    ws.append([None, 2])
    ws.append([None, 3])
    wb.save(path)
    wb.close()

    report = forward_fill_athlete_suggestions(
        path,
        "Sheet1",
        header_rows=[0, 1],
        # Имя плоской колонки совпадает со схемой mapping.flatten_columns.
        athlete_columns=["Спортсмен | ФИО"],
    )
    assert report.athlete_column == "Спортсмен | ФИО"
    rows = [(s.row, s.source_row, s.proposed) for s in report.suggestions]
    # header_rows=[0,1] → first data Excel row = 3.
    assert rows == [(4, 3, "Иванов"), (5, 3, "Иванов")]


def test_forward_fill_warns_when_athlete_columns_not_in_sheet(
    athlete_xlsx: Path,
) -> None:
    report = forward_fill_athlete_suggestions(
        athlete_xlsx,
        "Sheet1",
        header_rows=[0],
        athlete_columns=["Несуществующая колонка"],
    )
    assert report.suggestions == []
    assert report.athlete_column is None
    assert report.warning is not None
    assert "header_rows" in report.warning


def test_forward_fill_warns_when_no_athlete_role(athlete_xlsx: Path) -> None:
    report = forward_fill_athlete_suggestions(
        athlete_xlsx,
        "Sheet1",
        header_rows=[0],
        athlete_columns=[],
    )
    assert report.suggestions == []
    assert report.warning is not None
    assert "athlete" in report.warning


def test_forward_fill_warns_about_extra_athlete_columns(tmp_path: Path) -> None:
    path = tmp_path / "two_athletes.xlsx"
    # Соседняя колонка с числом нужна, чтобы пустая ФИО-строка вообще
    # сохранилась в xlsx (иначе openpyxl не запишет полностью пустую строку).
    df = pd.DataFrame(
        [
            ["Иванов", "Петрова", 1],
            [None, None, 2],
        ],
        columns=["ФИО борца", "ФИО соперника", "№ эпизода"],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)

    report = forward_fill_athlete_suggestions(
        path,
        "Sheet1",
        header_rows=[0],
        athlete_columns=["ФИО борца", "ФИО соперника"],
    )
    assert report.athlete_column == "ФИО борца"
    assert "ФИО соперника" in (report.warning or "")
    assert len(report.suggestions) == 1
    assert report.suggestions[0].col == 1


def test_forward_fill_skips_fully_empty_data_row(tmp_path: Path) -> None:
    path = tmp_path / "gaps.xlsx"
    df = pd.DataFrame(
        [
            ["Иванов", 1, 1],
            [None, None, None],
            [None, 2, 2],
        ],
        columns=["ФИО борца", "№ эпизода", "Время эпизода, с."],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Sheet1", index=False)
    report = forward_fill_athlete_suggestions(
        path,
        "Sheet1",
        header_rows=[0],
        athlete_columns=["ФИО борца"],
        episode_columns=["№ эпизода"],
    )
    rows = [s.row for s in report.suggestions]
    assert 3 not in rows
    assert 4 in rows


def test_forward_fill_with_episode_role_skips_row_without_episode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ep_only.xlsx"
    df = pd.DataFrame(
        [
            ["X", 1, 10],
            [None, 2, 20],
            [None, None, 30],
        ],
        columns=["ФИО борца", "№ эпизода", "Время эпизода, с."],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Sheet1", index=False)
    report = forward_fill_athlete_suggestions(
        path,
        "Sheet1",
        header_rows=[0],
        athlete_columns=["ФИО борца"],
        episode_columns=["№ эпизода"],
    )
    assert [s.row for s in report.suggestions] == [3]
    assert 4 not in [s.row for s in report.suggestions]


# --------------------------- header_merge_fill ------------------------------


def _build_merged_header_xlsx(path: Path) -> None:
    """Лист с двухстрочной шапкой и горизонтальным merge в верхней строке."""

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    # Row 1: "Спортсмен" покрывает A1:B1; C1 — самостоятельная "Эпизод"
    ws["A1"] = "Спортсмен"
    ws["C1"] = "Эпизод"
    ws.merge_cells("A1:B1")
    # Row 2: подзаголовки
    ws["A2"] = "ФИО"
    ws["B2"] = "Команда"
    ws["C2"] = "№"
    # Data
    ws["A3"] = "Иванов"
    ws["B3"] = "RUS"
    ws["C3"] = 1
    wb.save(path)
    wb.close()


def test_header_merge_fill_proposes_slave_cells(tmp_path: Path) -> None:
    path = tmp_path / "merged.xlsx"
    _build_merged_header_xlsx(path)
    report = header_merge_fill_suggestions(path, "Sheet1", header_rows=[0, 1])
    coords = [(s.row, s.col, s.proposed) for s in report.suggestions]
    # B1 — единственная slave-ячейка в merge A1:B1, master = "Спортсмен".
    assert coords == [(1, 2, "Спортсмен")]
    s0 = report.suggestions[0]
    assert s0.source_row == 1 and s0.source_col == 1


def test_header_merge_fill_skips_ranges_outside_header(tmp_path: Path) -> None:
    path = tmp_path / "out_of_header.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "H1"
    ws["B1"] = "H2"
    ws["A2"] = "v"
    ws["B2"] = "w"
    # merge ниже шапки — не должен попасть в предложения
    ws.merge_cells("A2:B2")
    wb.save(path)
    wb.close()

    report = header_merge_fill_suggestions(path, "Sheet1", header_rows=[0])
    assert report.suggestions == []


def test_header_merge_fill_skips_empty_master(tmp_path: Path) -> None:
    path = tmp_path / "empty_master.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    # master пустой — не предлагаем «вписать пустоту» в slave
    ws.merge_cells("A1:B1")
    ws["A2"] = "x"
    ws["B2"] = "y"
    wb.save(path)
    wb.close()

    report = header_merge_fill_suggestions(path, "Sheet1", header_rows=[0, 1])
    assert report.suggestions == []


def test_header_merge_fill_warns_on_empty_header_rows(tmp_path: Path) -> None:
    path = tmp_path / "no_header.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "x"
    wb.save(path)
    wb.close()
    report = header_merge_fill_suggestions(path, "Sheet1", header_rows=[])
    assert report.suggestions == []
    assert report.warning is not None
