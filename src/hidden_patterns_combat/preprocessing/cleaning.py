from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight MVP cleaning for episode tables.

    Keeps transformations transparent and reversible:
    - drops fully empty columns;
    - strips string values.
    """
    cleaned = df.copy()
    before_shape = cleaned.shape

    cleaned = cleaned.dropna(axis=1, how="all")
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    logger.info("Preprocessing: %s -> %s", before_shape, cleaned.shape)
    return cleaned
