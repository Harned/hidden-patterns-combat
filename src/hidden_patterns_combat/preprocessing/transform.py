from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_dictionary import DataDictionary
from .headers import normalize_columns


@dataclass
class TransformResult:
    cleaned: pd.DataFrame
    mapping: pd.DataFrame


def _classify_block(col: str, dd: DataDictionary) -> str:
    found = dd.lookup(col)
    if found:
        return found.logical_group
    return dd.infer_block(col)


def _canonical_name(col: str, block: str, position: int, dd: DataDictionary) -> str:
    found = dd.lookup(col)
    if found:
        return found.normalized_field if position == 1 else f"{found.normalized_field}_{position:02d}"

    if col in dd.canonical_metadata_map:
        base = dd.canonical_metadata_map[col]
        return base if position == 1 else f"{base}_{position:02d}"
    if block == "metadata":
        low = col.lower()
        if any(token in low for token in ("№ эпизода", "номер эпизода", "episode id", "episode number")):
            return "metadata__episode_id"
        if ("время" in low and "пауз" in low) or "pause duration" in low:
            return "metadata__pause_duration"
        if ("время" in low and "эпизод" in low) or "episode duration" in low:
            return "metadata__episode_duration"
        if "эпизод" in low:
            return f"metadata__episode_attr_{position:02d}"
        if "время" in low:
            return f"metadata__time_attr_{position:02d}"
    return f"{block}__f{position:02d}"


def _coerce_block_types(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_blocks = {"maneuvering", "kfv", "vup", "outcomes"}
    for _, row in mapping.iterrows():
        col = row["normalized_column"]
        block = row["block"]
        if col not in out.columns:
            continue

        if block in numeric_blocks:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = out[col].astype(str).replace("nan", "")

    return out


def transform_raw_to_tidy(raw_df: pd.DataFrame, data_dictionary: DataDictionary | None = None) -> TransformResult:
    dd = data_dictionary or DataDictionary.default()

    norm = normalize_columns(raw_df)
    mapping_rows: list[dict[str, object]] = []
    renamed: dict[str, str] = {}
    block_counts: dict[str, int] = {}

    for source_col in norm.columns:
        block = _classify_block(source_col, dd)
        block_counts[block] = block_counts.get(block, 0) + 1
        pos = block_counts[block]
        normalized = _canonical_name(source_col, block, pos, dd)

        while normalized in renamed.values():
            pos += 1
            normalized = _canonical_name(source_col, block, pos, dd)

        renamed[source_col] = normalized
        mapping_rows.append(
            {
                "source_column": source_col,
                "normalized_column": normalized,
                "block": block,
                "position_in_block": pos,
            }
        )

    cleaned = norm.rename(columns=renamed)
    cleaned = cleaned.dropna(axis=0, how="all").reset_index(drop=True)
    mapping = pd.DataFrame(mapping_rows)

    cleaned = _coerce_block_types(cleaned, mapping)
    return TransformResult(cleaned=cleaned, mapping=mapping)
