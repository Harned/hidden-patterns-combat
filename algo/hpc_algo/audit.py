"""Честный аудит Excel-источника.

Модуль отвечает только за структурные и статистические характеристики
данных: список листов, размеры, колонки, типы, пропуски, подозрительные
значения и preview. Никакой интерпретации скрытых состояний здесь нет.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from hpc_algo.loading import LoadedExcel
from hpc_algo.schema import AuditReport, ColumnInfo, SheetAudit

_PREVIEW_ROWS = 5
_SAMPLE_VALUES = 5

# Маркеры строк-итогов в листах судейской фиксации. Реальные файлы
# вида ``docs/Оценка СД содержание.xlsx`` иногда содержат хвостовые
# строки с агрегатами ("Итого", "Всего", "Сумма", "Среднее"), которые
# структурно неотличимы от обычной строки эпизода: те же числовые
# колонки заполнены, но это сумма, а не отдельное наблюдение. Если
# такие строки попадают в HMM, они кажутся аномально длинными
# эпизодами с экстремальными значениями ZAP.
_TOTALS_ROW_MARKERS: tuple[str, ...] = (
    "итого",
    "итог",
    "всего",
    "сумма",
    "среднее",
    "ср.",
    "total",
    "subtotal",
)


def _is_totals_row(row: pd.Series) -> bool:
    """True, если хотя бы одна строковая ячейка строки совпадает с маркером итогов.

    Сравнение нечувствительно к регистру и пробелам по краям. Только
    точное совпадение/префикс с маркером — иначе ложно сработали бы
    фамилии вроде «Итогов» или комментарии «итого получилось».
    """

    for value in row:
        if not isinstance(value, str):
            continue
        norm = value.strip().lower()
        if not norm:
            continue
        # Точное совпадение — самый надёжный сигнал.
        if norm in _TOTALS_ROW_MARKERS:
            return True
        # Префиксное совпадение со словом-маркером и разделителем (":" / " ").
        for marker in _TOTALS_ROW_MARKERS:
            if norm.startswith(marker + ":") or norm.startswith(marker + " "):
                return True
    return False


def detect_totals_row_indices(df: pd.DataFrame) -> list[int]:
    """Найти позиционные индексы (0..len-1) строк-итогов в df.

    Возвращает позиции строк, а не значения ``df.index`` — это нужно
    чтобы downstream-код (drop_totals_rows) был устойчив к
    переиндексациям, делаемым предыдущими шагами пайплайна.
    """

    if df.empty:
        return []
    return [pos for pos, (_, row) in enumerate(df.iterrows()) if _is_totals_row(row)]


def drop_totals_rows(df: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    """Вернуть копию df без строк по указанным позициям (0..len-1).

    Если ``positions`` пусто — df возвращается as-is (без копии).
    """

    if not positions:
        return df
    keep_mask = np.ones(len(df), dtype=bool)
    valid = [p for p in positions if 0 <= p < len(df)]
    if not valid:
        return df
    keep_mask[valid] = False
    return df.iloc[keep_mask].reset_index(drop=True)


def _stringify(value: Any) -> Any:
    """JSON-совместимая сериализация значения ячейки."""

    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sample_unique(series: pd.Series, limit: int = _SAMPLE_VALUES) -> list[Any]:
    """Вернуть до ``limit`` уникальных непустых значений."""

    dropped = series.dropna()
    if dropped.empty:
        return []
    # drop_duplicates сохраняет порядок появления — это полезнее random sample.
    uniques = dropped.drop_duplicates().head(limit).tolist()
    return [_stringify(v) for v in uniques]


def _suspicious_observations(df: pd.DataFrame) -> list[str]:
    """Набор очень простых, но полезных флагов качества."""

    notes: list[str] = []

    if df.empty:
        notes.append("Лист пустой.")
        return notes

    # Подозрение на многострочный заголовок: большая часть колонок не имеет
    # осмысленного имени (pandas даёт 'Unnamed: N', когда первая строка — пустая
    # или часть merged-заголовка).
    cols = [str(c) for c in df.columns]
    unnamed = [c for c in cols if c.startswith("Unnamed:")]
    if cols and len(unnamed) / len(cols) >= 0.5:
        notes.append(
            f"Большинство колонок без имён ({len(unnamed)}/{len(cols)}). "
            "Похоже на многострочный заголовок — см. первые строки preview: реальные "
            "подзаголовки могут находиться в строках 1–2 данных."
        )

    # Полностью пустые колонки.
    all_null_cols = [str(c) for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        notes.append(
            "Полностью пустые колонки: "
            + ", ".join(all_null_cols[:10])
            + ("..." if len(all_null_cols) > 10 else "")
        )

    # Полностью пустые строки.
    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        notes.append(f"Полностью пустых строк: {empty_rows}.")

    # Дубликаты заголовков (pandas добавляет .1, .2 и т.п., ловим по исходному имени).
    cols = [str(c) for c in df.columns]
    dup_headers: dict[str, int] = {}
    for c in cols:
        base = c.rsplit(".", 1)[0] if "." in c and c.rsplit(".", 1)[1].isdigit() else c
        dup_headers[base] = dup_headers.get(base, 0) + 1
    dups = [k for k, v in dup_headers.items() if v > 1]
    if dups:
        notes.append("Возможные дубликаты заголовков: " + ", ".join(dups[:10]))

    # Смешанные типы данных в object-колонках.
    mixed_cols: list[str] = []
    for c in df.columns:
        if df[c].dtype == object:
            kinds = {type(v).__name__ for v in df[c].dropna().head(50)}
            if len(kinds) > 1:
                mixed_cols.append(f"{c}: {sorted(kinds)}")
    if mixed_cols:
        notes.append("Смешанные типы значений: " + "; ".join(mixed_cols[:5]))

    return notes


def _audit_sheet(name: str, df: pd.DataFrame) -> SheetAudit:
    n_rows, n_cols = df.shape

    columns: list[ColumnInfo] = []
    for raw_col in df.columns:
        col = str(raw_col)
        series = df[raw_col]
        non_null = int(series.notna().sum())
        null = int(series.isna().sum())
        total = non_null + null
        null_ratio = (null / total) if total else 0.0
        columns.append(
            ColumnInfo(
                name=col,
                dtype=str(series.dtype),
                non_null_count=non_null,
                null_count=null,
                null_ratio=round(null_ratio, 4),
                unique_count=int(series.nunique(dropna=True)),
                sample_values=_sample_unique(series),
            )
        )

    preview_rows: list[dict[str, Any]] = []
    for _, row in df.head(_PREVIEW_ROWS).iterrows():
        preview_rows.append({str(k): _stringify(v) for k, v in row.items()})

    totals_positions = detect_totals_row_indices(df)
    suspicious = _suspicious_observations(df)
    if totals_positions:
        suspicious.append(
            f"Распознаны строки-итоги (n={len(totals_positions)}). "
            "Они будут исключены из baseline/HMM, чтобы суммы не "
            "интерпретировались как отдельные эпизоды."
        )

    return SheetAudit(
        name=name,
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        columns=columns,
        preview=preview_rows,
        suspicious=suspicious,
        totals_row_indices=totals_positions,
    )


def build_audit(loaded: LoadedExcel) -> AuditReport:
    """Собрать :class:`AuditReport` для загруженного Excel."""

    sheet_reports = [_audit_sheet(name, df) for name, df in loaded.sheets.items()]
    return _aggregate_audit(sheet_reports)


def filter_audit_to_sheets(audit: AuditReport, names: set[str]) -> AuditReport:
    """Сузить :class:`AuditReport` к подмножеству листов.

    Используется в mapping-ветке, чтобы `data_audit` и сводка в текстовом
    отчёте не описывали листы, которые пользователь исключил из анализа.
    Порядок листов сохраняется как в исходном `audit.sheets`. Агрегаты
    (`total_rows`, `total_cells`, `overall_null_ratio`) пересчитываются по
    оставшимся листам по той же формуле, что и в :func:`build_audit`.
    """

    if not names:
        return _aggregate_audit([])
    selected = [s for s in audit.sheets if s.name in names]
    return _aggregate_audit(selected)


def _aggregate_audit(sheet_reports: list[SheetAudit]) -> AuditReport:
    total_rows = sum(s.n_rows for s in sheet_reports)
    total_cells = sum(s.n_rows * s.n_cols for s in sheet_reports)

    if total_cells:
        total_nulls = sum(
            col.null_count for s in sheet_reports for col in s.columns
        )
        overall_null_ratio = total_nulls / total_cells
    else:
        overall_null_ratio = 0.0

    return AuditReport(
        sheets=sheet_reports,
        total_rows=total_rows,
        total_cells=total_cells,
        overall_null_ratio=round(overall_null_ratio, 4),
    )
