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

    if save_parquet:
        try:
            raw_parquet = out / "raw_combined.parquet"
            cleaned_parquet = out / "cleaned_tidy.parquet"
            raw_df.to_parquet(raw_parquet, index=False)
            cleaned_df.to_parquet(cleaned_parquet, index=False)
            result["raw_parquet"] = str(raw_parquet)
            result["cleaned_parquet"] = str(cleaned_parquet)
        except Exception:
            # Optional output: keep MVP robust without hard dependency on parquet engine.
            pass

    return result
