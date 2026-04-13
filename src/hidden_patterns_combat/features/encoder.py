from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from hidden_patterns_combat.config import FeatureConfig
from hidden_patterns_combat.features.engineering import (
    FeatureEngineer,
    FeatureEngineeringConfig,
    FeatureValidationReport,
)


@dataclass
class EncodedBatch:
    raw: pd.DataFrame
    features: pd.DataFrame
    metadata: pd.DataFrame
    traceability: pd.DataFrame
    validation: FeatureValidationReport


def encode_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    engineering_cfg: FeatureEngineeringConfig | None = None,
) -> EncodedBatch:
    """Backward-compatible facade for feature engineering layer.

    Accepts cleaned DataFrame and returns engineered features for modeling.
    """
    result = FeatureEngineer(cfg, engineering_cfg=engineering_cfg).transform(df)
    return EncodedBatch(
        raw=result.raw_feature_set,
        features=result.engineered_feature_set,
        metadata=result.metadata,
        traceability=result.traceability,
        validation=result.validation,
    )
