from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from hidden_patterns_combat.config import FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class EncodedBatch:
    raw: pd.DataFrame
    features: pd.DataFrame
    metadata: pd.DataFrame


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    as_str = series.astype(str).str.strip().str.lower()
    mapper = {
        "1": 1,
        "0": 0,
        "да": 1,
        "нет": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
    }
    mapped = as_str.map(mapper)
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.where(~mapped.isna(), (numeric > 0).astype(float))
    return mapped.fillna(0).astype(int)


def _find_columns(df: pd.DataFrame, tokens: Iterable[str]) -> list[str]:
    tokens_norm = [t.lower() for t in tokens]
    result: list[str] = []
    for col in df.columns:
        low = col.lower()
        if any(token in low for token in tokens_norm):
            result.append(col)
    return result


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
        for col in df.columns:
            if candidate in col.lower():
                return col
    return None


def _split_maneuver_columns_for_mvp(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[list[str], list[str]]:
    right_cols = _find_columns(df, cfg.maneuver_right_tokens)
    left_cols = _find_columns(df, cfg.maneuver_left_tokens)
    if right_cols or left_cols:
        return right_cols, left_cols

    # Fallback for wide tables where right/left subheaders are lost after Excel import:
    # split the maneuver block into two halves to preserve the research compact encoding.
    group_cols = _find_columns(df, cfg.maneuver_group_tokens)
    if not group_cols:
        return [], []

    midpoint = len(group_cols) // 2
    if midpoint == 0:
        return group_cols, []
    return group_cols[:midpoint], group_cols[midpoint:]


def _compact_code(binary_frame: pd.DataFrame) -> pd.Series:
    if binary_frame.empty:
        return pd.Series(np.zeros(len(binary_frame), dtype=int), index=binary_frame.index)

    bits = np.array([1 << i for i in range(binary_frame.shape[1])], dtype=np.int64)
    data = binary_frame.to_numpy(dtype=np.int64)
    return pd.Series((data * bits).sum(axis=1), index=binary_frame.index)


def encode_features(df: pd.DataFrame, cfg: FeatureConfig) -> EncodedBatch:
    df = df.copy()

    right_cols, left_cols = _split_maneuver_columns_for_mvp(df, cfg)
    kfv_cols = _find_columns(df, cfg.kfv_tokens)
    vup_cols = _find_columns(df, cfg.vup_tokens)

    logger.info(
        "Found maneuver_right=%d, maneuver_left=%d, kfv=%d, vup=%d columns",
        len(right_cols), len(left_cols), len(kfv_cols), len(vup_cols),
    )

    encoded = pd.DataFrame(index=df.index)
    encoded["maneuver_right_code"] = _compact_code(df[right_cols].apply(_to_binary) if right_cols else pd.DataFrame(index=df.index))
    encoded["maneuver_left_code"] = _compact_code(df[left_cols].apply(_to_binary) if left_cols else pd.DataFrame(index=df.index))
    encoded["kfv_code"] = _compact_code(df[kfv_cols].apply(_to_binary) if kfv_cols else pd.DataFrame(index=df.index))
    encoded["vup_code"] = _compact_code(df[vup_cols].apply(_to_binary) if vup_cols else pd.DataFrame(index=df.index))

    duration_col = _first_existing(df, cfg.duration_column_candidates)
    pause_col = _first_existing(df, cfg.pause_column_candidates)
    result_col = _first_existing(df, cfg.result_column_candidates)
    episode_col = _first_existing(df, cfg.episode_id_column_candidates)

    encoded["duration"] = pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0) if duration_col else 0.0
    encoded["pause"] = pd.to_numeric(df[pause_col], errors="coerce").fillna(0.0) if pause_col else 0.0
    encoded["observed_result"] = pd.to_numeric(df[result_col], errors="coerce").fillna(0.0) if result_col else 0.0

    metadata = pd.DataFrame(index=df.index)
    metadata["episode_id"] = df[episode_col].astype(str) if episode_col else pd.Series(df.index.astype(str), index=df.index)

    return EncodedBatch(raw=df, features=encoded, metadata=metadata)
