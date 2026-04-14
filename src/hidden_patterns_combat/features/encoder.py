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


DEFAULT_HMM_FEATURE_ORDER: tuple[str, ...] = (
    "maneuver_right_code",
    "maneuver_left_code",
    "grips_code",
    "holds_code",
    "bodylocks_code",
    "underhooks_code",
    "posts_code",
    "kfv_code",
    "vup_code",
    "duration",
    "pause",
)


def select_hmm_input_features(engineered_features: pd.DataFrame) -> pd.DataFrame:
    """Select HMM input features without target/outcome leakage.

    Outcome/result columns are intentionally excluded from model input and can be
    used only for post-hoc interpretation/diagnostics.
    """
    ordered = [c for c in DEFAULT_HMM_FEATURE_ORDER if c in engineered_features.columns]
    if ordered:
        return engineered_features[ordered].copy()

    fallback = [
        c
        for c in engineered_features.columns
        if "outcome" not in c.lower()
        and "result" not in c.lower()
        and "score" not in c.lower()
    ]
    return engineered_features[fallback].copy()


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
