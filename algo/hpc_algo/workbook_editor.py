"""Редактирование Excel-файла на стороне processing module.

Любая запись в xlsx должна происходить здесь, не в backend-роутах и не
в UI. Модуль использует ``openpyxl``, чтобы не пересобирать workbook
целиком (что разрушило бы стили, merged-ячейки и порядок листов).

Поддерживает:

* :func:`read_grid` — чтение прямоугольного блока «как видит Excel»
  (1-based строки) для виртуализированной таблицы UI;
* :func:`apply_cell_edits` — точечное обновление ячеек;
* :func:`remove_empty_rows` — удаление полностью пустых data-строк
  с учётом подтверждённого мастером ``header_rows``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.cell.cell import MergedCell
from openpyxl.utils import get_column_letter


def _to_jsonable(value: Any) -> Any:
    """Подготовить значение Excel-ячейки к сериализации."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _ensure_writable_cell(ws, row: int, col: int):
    """Для :class:`MergedCell` (все ячейки кроме master) — **снять** merge с
    диапазоном, чтобы можно было писать в *эту* строку. Иначе перенос в
    master (часто нижняя часть объединения) портит смысл «ФИО к эпизодам».
    """

    c = ws.cell(row=row, column=col)
    if not isinstance(c, MergedCell):
        return c
    for mrange in list(ws.merged_cells.ranges):
        if mrange.min_row <= row <= mrange.max_row and mrange.min_col <= col <= mrange.max_col:
            a1 = (
                f"{get_column_letter(mrange.min_col)}{mrange.min_row}"
                f":{get_column_letter(mrange.max_col)}{mrange.max_row}"
            )
            ws.unmerge_cells(a1)
            return ws.cell(row=row, column=col)
    return c


def _coerce_for_excel(value: Any) -> Any:
    """Привести входное значение из UI к тому, что openpyxl сохранит как ячейку."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if text == "":
        return None
    return text


@dataclass(frozen=True)
class GridFragment:
    sheet: str
    start_row: int
    start_col: int
    n_rows: int
    n_cols: int
    total_rows: int
    total_cols: int
    cells: list[list[Any]]


def read_grid(
    path: str | Path,
    sheet_name: str,
    *,
    start_row: int = 1,
    start_col: int = 1,
    n_rows: int = 100,
    n_cols: int = 50,
) -> GridFragment:
    """Прочитать прямоугольный блок ячеек ``sheet_name``.

    Координаты — 1-based, как в Excel. Если запрошен фрагмент за
    пределами листа, возвращаются ``None`` для отсутствующих ячеек.
    """

    if start_row < 1 or start_col < 1 or n_rows < 0 or n_cols < 0:
        raise ValueError("start_row/start_col >= 1, n_rows/n_cols >= 0")

    wb = load_workbook(filename=str(path), data_only=True, read_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            raise KeyError(f"Лист '{sheet_name}' не найден.")
        ws = wb[sheet_name]
        total_rows = int(ws.max_row or 0)
        total_cols = int(ws.max_column or 0)

        end_row = start_row + n_rows - 1
        end_col = start_col + n_cols - 1
        cells: list[list[Any]] = []
        if n_rows == 0 or n_cols == 0:
            return GridFragment(
                sheet=sheet_name,
                start_row=start_row,
                start_col=start_col,
                n_rows=0,
                n_cols=0,
                total_rows=total_rows,
                total_cols=total_cols,
                cells=[],
            )

        rows_iter = ws.iter_rows(
            min_row=start_row,
            max_row=end_row,
            min_col=start_col,
            max_col=end_col,
            values_only=True,
        )
        for row_values in rows_iter:
            row: list[Any] = [_to_jsonable(v) for v in row_values]
            while len(row) < n_cols:
                row.append(None)
            cells.append(row)
        # Дополним пустыми строками, если лист короче.
        while len(cells) < n_rows:
            cells.append([None] * n_cols)
    finally:
        wb.close()

    return GridFragment(
        sheet=sheet_name,
        start_row=start_row,
        start_col=start_col,
        n_rows=n_rows,
        n_cols=n_cols,
        total_rows=total_rows,
        total_cols=total_cols,
        cells=cells,
    )


@dataclass(frozen=True)
class CellEdit:
    row: int
    col: int
    value: Any


def apply_cell_edits(
    path: str | Path,
    sheet_name: str,
    edits: list[CellEdit] | list[dict[str, Any]],
) -> int:
    """Применить точечные правки ячеек к листу и сохранить файл.

    Координаты ``row``/``col`` — 1-based. Возвращает количество ячеек,
    которые фактически были обновлены.
    """

    normalized: list[CellEdit] = []
    for e in edits:
        if isinstance(e, CellEdit):
            normalized.append(e)
            continue
        normalized.append(
            CellEdit(row=int(e["row"]), col=int(e["col"]), value=e.get("value"))
        )

    if not normalized:
        return 0

    wb = load_workbook(filename=str(path))
    try:
        if sheet_name not in wb.sheetnames:
            raise KeyError(f"Лист '{sheet_name}' не найден.")
        ws = wb[sheet_name]
        applied = 0
        for edit in normalized:
            if edit.row < 1 or edit.col < 1:
                raise ValueError(
                    f"Координаты ячейки должны быть >= 1, получено: {edit}"
                )
            cell = _ensure_writable_cell(ws, edit.row, edit.col)
            cell.value = _coerce_for_excel(edit.value)
            applied += 1
        wb.save(str(path))
    finally:
        wb.close()
    return applied


def count_empty_rows(
    path: str | Path,
    sheet_name: str,
    *,
    header_rows: list[int] | None = None,
) -> int:
    """Сосчитать data-строки, у которых все ячейки пустые.

    Использует ту же логику, что и :func:`remove_empty_rows`, но открывает
    workbook в режиме ``read_only`` и не модифицирует файл, чтобы UI мог
    дешёво показать пользователю «обнаружено N пустых строк».
    """

    wb = load_workbook(filename=str(path), data_only=True, read_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            raise KeyError(f"Лист '{sheet_name}' не найден.")
        ws = wb[sheet_name]

        header_set = set(header_rows or [0])
        first_data_row_1based = (max(header_set) + 1) + 1
        max_row = int(ws.max_row or 0)
        if max_row < first_data_row_1based:
            return 0
        max_col = int(ws.max_column or 0)
        if max_col < 1:
            return 0

        # Один проход по диапазону: отдельный iter_rows на каждую строку
        # на крупных листах (1000+ строк) даёт минуты ожидания в UI.
        count = 0
        for row_values in ws.iter_rows(
            min_row=first_data_row_1based,
            max_row=max_row,
            min_col=1,
            max_col=max_col,
            values_only=True,
        ):
            if all(
                v is None or (isinstance(v, str) and v.strip() == "")
                for v in row_values
            ):
                count += 1
        return count
    finally:
        wb.close()


def remove_empty_rows(
    path: str | Path,
    sheet_name: str,
    *,
    header_rows: list[int] | None = None,
) -> int:
    """Удалить data-строки, у которых все ячейки пустые.

    ``header_rows`` — 0-based индексы заголовочных строк (как в
    :class:`hpc_algo.schema.SheetMapping`). Они не трогаются. Возвращает
    количество удалённых строк.
    """

    wb = load_workbook(filename=str(path))
    try:
        if sheet_name not in wb.sheetnames:
            raise KeyError(f"Лист '{sheet_name}' не найден.")
        ws = wb[sheet_name]

        header_set = set(header_rows or [0])
        first_data_row_1based = (max(header_set) + 1) + 1

        to_delete: list[int] = []
        max_r = int(ws.max_row or 0)
        max_c = int(ws.max_column or 0)
        if max_c < 1 or max_r < first_data_row_1based:
            return 0
        for row_idx, row_values in enumerate(
            ws.iter_rows(
                min_row=first_data_row_1based,
                max_row=max_r,
                min_col=1,
                max_col=max_c,
                values_only=True,
            ),
            start=first_data_row_1based,
        ):
            if all(
                v is None or (isinstance(v, str) and v.strip() == "")
                for v in row_values
            ):
                to_delete.append(row_idx)

        # Удаляем снизу вверх, чтобы индексы не сдвигались.
        for row_idx in reversed(to_delete):
            ws.delete_rows(row_idx, 1)

        if to_delete:
            wb.save(str(path))
        return len(to_delete)
    finally:
        wb.close()


__all__ = [
    "CellEdit",
    "GridFragment",
    "apply_cell_edits",
    "count_empty_rows",
    "read_grid",
    "remove_empty_rows",
]
