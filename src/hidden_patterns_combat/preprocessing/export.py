from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .validation import ValidationReport


def export_preprocessing_outputs(
    output_dir: str | Path,
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    validation: ValidationReport,
    save_parquet: bool = False,
    episodes_tidy_df: pd.DataFrame | None = None,
    matrix_label_mapping_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_csv = out / "raw_combined.csv"
    cleaned_csv = out / "cleaned_tidy.csv"
    mapping_csv = out / "data_dictionary.csv"
    validation_json = out / "validation.json"

    raw_df.to_csv(raw_csv, index=False)
    cleaned_df.to_csv(cleaned_csv, index=False)
    mapping_df.to_csv(mapping_csv, index=False)
    validation_json.write_text(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "raw_csv": str(raw_csv),
        "cleaned_csv": str(cleaned_csv),
        "mapping_csv": str(mapping_csv),
        "validation_json": str(validation_json),
    }

    if episodes_tidy_df is not None and not episodes_tidy_df.empty:
        episodes_tidy_csv = out / "episodes_tidy.csv"
        episodes_tidy_df.to_csv(episodes_tidy_csv, index=False)
        result["episodes_tidy_csv"] = str(episodes_tidy_csv)

    if matrix_label_mapping_df is not None and not matrix_label_mapping_df.empty:
        matrix_mapping_csv = out / "matrix_label_mapping.csv"
        matrix_label_mapping_df.to_csv(matrix_mapping_csv, index=False)
        result["matrix_label_mapping_csv"] = str(matrix_mapping_csv)

    if save_parquet:
        try:
            raw_parquet = out / "raw_combined.parquet"
            cleaned_parquet = out / "cleaned_tidy.parquet"
            raw_df.to_parquet(raw_parquet, index=False)
            cleaned_df.to_parquet(cleaned_parquet, index=False)
            result["raw_parquet"] = str(raw_parquet)
            result["cleaned_parquet"] = str(cleaned_parquet)
            if episodes_tidy_df is not None and not episodes_tidy_df.empty:
                episodes_tidy_parquet = out / "episodes_tidy.parquet"
                episodes_tidy_df.to_parquet(episodes_tidy_parquet, index=False)
                result["episodes_tidy_parquet"] = str(episodes_tidy_parquet)
        except Exception:
            # Optional output: keep MVP robust without hard dependency on parquet engine.
            pass

    return result
