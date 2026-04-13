from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExcelSheet:
    name: str
    dataframe: pd.DataFrame


def _normalize_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text.startswith("unnamed"):
        return ""
    return " ".join(text.split())


def flatten_columns(columns: pd.Index | pd.MultiIndex) -> list[str]:
    normalized: list[str] = []
    if isinstance(columns, pd.MultiIndex):
        for col in columns:
            tokens = [_normalize_token(part) for part in col]
            tokens = [t for t in tokens if t]
            normalized.append(" | ".join(tokens) if tokens else "unknown_column")
    else:
        normalized = [_normalize_token(c) or "unknown_column" for c in columns]

    seen: dict[str, int] = {}
    unique: list[str] = []
    for col in normalized:
        count = seen.get(col, 0)
        seen[col] = count + 1
        unique.append(f"{col}_{count + 1}" if count else col)
    return unique


def read_excel_sheets(
    excel_path: str | Path,
    sheets: Iterable[str] | None = None,
    header_depth: int = 2,
) -> list[ExcelSheet]:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    header = list(range(header_depth)) if header_depth > 1 else 0
    sheet_map = pd.read_excel(excel_path, sheet_name=None, header=header, engine="openpyxl")

    selected = set(sheets) if sheets else None
    result: list[ExcelSheet] = []

    for sheet_name, df in sheet_map.items():
        if selected and sheet_name not in selected:
            continue
        df = df.copy()
        df.columns = flatten_columns(df.columns)
        df = df.dropna(axis=0, how="all").reset_index(drop=True)
        logger.info("Loaded sheet '%s' with shape %s", sheet_name, df.shape)
        result.append(ExcelSheet(name=sheet_name, dataframe=df))

    if not result:
        raise ValueError("No sheets loaded. Check sheet names and workbook content.")

    return result
