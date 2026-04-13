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

    lengths: list[int] = []
    current = sequence_ids.iloc[0]
    count = 0
    for value in sequence_ids:
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
) -> ObservationBatch:
    x_raw = features.to_numpy(dtype=float)
    x = scaler.fit_transform(x_raw) if fit_scaler else scaler.transform(x_raw)
    lengths = build_lengths(sequence_ids)
    return ObservationBatch(values=x, feature_columns=list(features.columns), lengths=lengths)
