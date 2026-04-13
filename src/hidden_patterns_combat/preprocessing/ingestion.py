from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from hidden_patterns_combat.io.excel_loader import flatten_columns


SheetSelector = str | Iterable[str] | None


@dataclass
class IngestionResult:
    sheets_loaded: list[str]
    raw_combined: pd.DataFrame


def _resolve_sheet_names(excel_path: Path, selector: SheetSelector) -> list[str]:
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    all_names = list(xls.sheet_names)

    if selector is None:
        return all_names
    if isinstance(selector, str):
        selected = [selector]
    else:
        selected = list(selector)

    unknown = [s for s in selected if s not in all_names]
    if unknown:
        raise ValueError(f"Unknown sheet(s): {unknown}. Available: {all_names}")
    return selected


def load_excel_for_preprocessing(
    excel_path: str | Path,
    sheet_selector: SheetSelector = None,
    header_depth: int = 2,
) -> IngestionResult:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    selected = _resolve_sheet_names(path, sheet_selector)
    header = list(range(header_depth)) if header_depth > 1 else 0

    frames: list[pd.DataFrame] = []
    for sheet in selected:
        df = pd.read_excel(path, sheet_name=sheet, header=header, engine="openpyxl")
        df.columns = flatten_columns(df.columns)
        df = df.dropna(axis=0, how="all").reset_index(drop=True)
        df["_sheet"] = sheet
        frames.append(df)

    combined = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    return IngestionResult(sheets_loaded=selected, raw_combined=combined)
