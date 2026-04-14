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


def select_hmm_input_features(
    engineered_features: pd.DataFrame,
    *,
    return_info: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, list[str]]]:
    """Select HMM input features without target/outcome leakage.

    Outcome/result columns are intentionally excluded from model input and can be
    used only for post-hoc interpretation/diagnostics.
    """
    ordered = [c for c in DEFAULT_HMM_FEATURE_ORDER if c in engineered_features.columns]
    selected = engineered_features[ordered].copy() if ordered else engineered_features.copy()
    fallback = [
        c
        for c in selected.columns
        if "outcome" not in c.lower()
        and "result" not in c.lower()
        and "score" not in c.lower()
    ]
    selected = selected[fallback].copy()

    numeric = selected.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    variances = numeric.var(axis=0, ddof=0)
    informative_cols = [c for c in numeric.columns if float(variances[c]) > 1e-12]
    dropped_constant = [c for c in numeric.columns if c not in informative_cols]

    if not informative_cols:
        informative_cols = list(numeric.columns)
        dropped_constant = []

    hmm_input = numeric[informative_cols].copy()
    info = {
        "hmm_input_features": informative_cols,
        "dropped_constant_features": dropped_constant,
    }
    if return_info:
        return hmm_input, info
    return hmm_input


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
