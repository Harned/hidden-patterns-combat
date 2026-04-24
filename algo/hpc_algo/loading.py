"""Чтение Excel-источника без какой-либо бизнес-логики.

Модуль сознательно узкий: принимает путь к файлу, возвращает словарь
``{sheet_name: DataFrame}`` и сырой список листов. Никакого переименования
колонок, нормализации или интерпретации содержимого — это работа audit/detection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LoadedExcel:
    """Результат чтения Excel-файла."""

    path: Path
    size_bytes: int
    sha256: str
    sheet_names: list[str]
    sheets: dict[str, pd.DataFrame]


class ExcelLoadError(RuntimeError):
    """Проблема при чтении Excel, которая должна быть отражена в ``errors``."""


def load_excel(path: str | Path) -> LoadedExcel:
    """Прочитать Excel-файл.

    * читает все листы через ``pandas.read_excel(..., sheet_name=None)``;
    * считает sha256 от содержимого;
    * не делает никаких предположений о структуре колонок;
    * при проблемах поднимает :class:`ExcelLoadError`.
    """

    p = Path(path)
    if not p.exists():
        raise ExcelLoadError(f"Файл не найден: {p}")
    if not p.is_file():
        raise ExcelLoadError(f"Путь не является файлом: {p}")

    suffix = p.suffix.lower()
    if suffix not in {".xlsx", ".xls", ".xlsm"}:
        raise ExcelLoadError(
            f"Неподдерживаемое расширение: {suffix}. Ожидается .xlsx, .xls или .xlsm."
        )

    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ExcelLoadError(f"Не удалось прочитать файл {p}: {exc}") from exc

    sha = hashlib.sha256(raw).hexdigest()

    try:
        sheets = pd.read_excel(p, sheet_name=None, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 — хотим показать пользователю сырую причину
        raise ExcelLoadError(f"pandas.read_excel не смог открыть {p}: {exc}") from exc

    # Сохраним порядок листов, возвращённый pandas.
    sheet_names = list(sheets.keys())

    return LoadedExcel(
        path=p,
        size_bytes=len(raw),
        sha256=sha,
        sheet_names=sheet_names,
        sheets=sheets,
    )
