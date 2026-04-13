from __future__ import annotations

import re

import pandas as pd


def normalize_column_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\|+", "|", text)
    return text


def deduplicate_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        base = col or "unknown_column"
        cnt = seen.get(base, 0)
        seen[base] = cnt + 1
        out.append(base if cnt == 0 else f"{base}_{cnt + 1}")
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [normalize_column_token(c) for c in out.columns]
    out.columns = deduplicate_columns(cols)
    return out
