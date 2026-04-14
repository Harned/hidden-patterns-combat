from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class ObservationBatch:
    values: np.ndarray
    feature_columns: list[str]
    lengths: list[int]


def build_lengths(sequence_ids: pd.Series | None) -> list[int]:
    if sequence_ids is None:
        return []
    if len(sequence_ids) == 0:
        return []

    normalized = (
        sequence_ids.fillna("sequence_0")
        .astype(str)
        .str.strip()
        .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
    )

    lengths: list[int] = []
    current = normalized.iloc[0]
    count = 1
    for value in normalized.iloc[1:]:
        if value == current:
            count += 1
        else:
            lengths.append(count)
            current = value
            count = 1
    lengths.append(count)
    return lengths


def encode_observations(
    features: pd.DataFrame,
    scaler: StandardScaler,
    fit_scaler: bool,
    sequence_ids: pd.Series | None = None,
    post_scale_weights: np.ndarray | None = None,
) -> ObservationBatch:
    x_raw = features.to_numpy(dtype=float)
    x = scaler.fit_transform(x_raw) if fit_scaler else scaler.transform(x_raw)
    if post_scale_weights is not None:
        if len(post_scale_weights) != x.shape[1]:
            raise ValueError(
                "post_scale_weights shape mismatch: "
                f"weights={len(post_scale_weights)} features={x.shape[1]}"
            )
        x = x * post_scale_weights.reshape(1, -1)
    lengths = build_lengths(sequence_ids)
    return ObservationBatch(values=x, feature_columns=list(features.columns), lengths=lengths)
