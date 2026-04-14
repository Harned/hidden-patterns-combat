from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


_AGG_TOKENS = ("итог", "сумма", "всего", "total", "summary")


def _first_existing(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
        for col in df.columns:
            if candidate in col.lower():
                return col
    return None


def _activity_series(df: pd.DataFrame) -> pd.Series:
    activity_cols = [
        c
        for c in df.columns
        if any(token in c.lower() for token in ("maneuver", "манев", "kfv", "контак", "vup", "выведение", "outcome", "заверша"))
    ]
    if not activity_cols:
        return pd.Series([0.0] * len(df), index=df.index)
    block = df[activity_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return block.abs().sum(axis=1)


def clean_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """MVP cleaning for episode tables with cautious filtering.

    Rules:
    - remove fully empty columns;
    - strip textual values;
    - remove fully empty rows;
    - remove obvious aggregate rows by athlete/meta text tokens;
    - remove rows with no episode id, no observed result, and no feature activity.
    """
    cleaned = df.copy()
    before_shape = cleaned.shape

    cleaned = cleaned.dropna(axis=1, how="all")
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    athlete_col = _first_existing(cleaned, ("metadata__athlete_name", "фио борца", "athlete_name"))
    sheet_col = _first_existing(cleaned, ("metadata__sheet", "_sheet", "sheet"))
    if athlete_col:
        cleaned[athlete_col] = (
            cleaned[athlete_col]
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
            .ffill()
        )
    if sheet_col:
        cleaned[sheet_col] = (
            cleaned[sheet_col]
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
            .ffill()
        )

    empty_rows_mask = cleaned.replace("", pd.NA).isna().all(axis=1)
    dropped_empty_rows = int(empty_rows_mask.sum())
    cleaned = cleaned.loc[~empty_rows_mask].copy()

    episode_col = _first_existing(
        cleaned,
        ("metadata__episode_id", "номер эпизода", "episode_id", "metadata__episode_attr_01", "эпизод"),
    )
    result_col = _first_existing(cleaned, ("outcomes__score", "баллы", "observed_result", "результат"))

    dropped_aggregate_rows = 0
    if athlete_col:
        athlete_text = cleaned[athlete_col].astype(str).str.lower().str.strip()
        aggregate_mask = athlete_text.str.contains("|".join(_AGG_TOKENS), regex=True, na=False)
        dropped_aggregate_rows = int(aggregate_mask.sum())
        cleaned = cleaned.loc[~aggregate_mask].copy()

    activity = _activity_series(cleaned)
    if episode_col:
        episode_text = cleaned[episode_col].astype(str).str.strip()
        episode_numeric = pd.to_numeric(cleaned[episode_col], errors="coerce")
        id_like = episode_text.str.fullmatch(r"[a-zA-Zа-яА-Я]*\d+[a-zA-Zа-яА-Я]*", na=False)
        looks_like_header = episode_text.str.lower().str.contains("эпизод|episode", regex=True, na=False)
        has_episode = (episode_numeric.notna() | id_like) & (~looks_like_header)
    else:
        has_episode = pd.Series([False] * len(cleaned), index=cleaned.index)

    if result_col:
        result_numeric = pd.to_numeric(cleaned[result_col], errors="coerce").fillna(0.0)
        has_result = result_numeric.ne(0.0)
    else:
        has_result = pd.Series([False] * len(cleaned), index=cleaned.index)

    low_info_mask = (~has_episode) & (~has_result) & (activity <= 0)
    dropped_low_info_rows = int(low_info_mask.sum())
    cleaned = cleaned.loc[~low_info_mask].copy()

    cleaned = cleaned.reset_index(drop=True)

    logger.info(
        "Preprocessing cleaning: %s -> %s | dropped_empty_rows=%d | dropped_aggregate_rows=%d | dropped_low_info_rows=%d",
        before_shape,
        cleaned.shape,
        dropped_empty_rows,
        dropped_aggregate_rows,
        dropped_low_info_rows,
    )
    return cleaned
