from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from hidden_patterns_combat.io.excel_loader import read_excel_sheets


SheetSelector = str | Iterable[str] | None


@dataclass
class IngestionResult:
    sheets_loaded: list[str]
    raw_combined: pd.DataFrame
    episodes_tidy: pd.DataFrame | None = None
    matrix_label_mapping: pd.DataFrame | None = None


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
    force_matrix_parser: bool = False,
) -> IngestionResult:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    selected = _resolve_sheet_names(path, sheet_selector)
    loaded = read_excel_sheets(
        excel_path=path,
        sheets=selected,
        header_depth=header_depth,
        force_matrix_parser=force_matrix_parser,
    )

    frames: list[pd.DataFrame] = []
    matrix_frames: list[pd.DataFrame] = []
    mapping_frames: list[pd.DataFrame] = []
    for sheet in loaded:
        df = sheet.dataframe.copy()
        if "_sheet" not in df.columns:
            df["_sheet"] = sheet.name
        frames.append(df)

        if sheet.parser_type == "matrix":
            matrix_frames.append(df.copy())
            if sheet.label_mapping is not None and not sheet.label_mapping.empty:
                mapping_frames.append(sheet.label_mapping.copy())

    combined = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    episodes_tidy = pd.concat(matrix_frames, axis=0, ignore_index=True) if matrix_frames else None
    mapping_df = pd.concat(mapping_frames, axis=0, ignore_index=True) if mapping_frames else None
    return IngestionResult(
        sheets_loaded=selected,
        raw_combined=combined,
        episodes_tidy=episodes_tidy,
        matrix_label_mapping=mapping_df,
    )
