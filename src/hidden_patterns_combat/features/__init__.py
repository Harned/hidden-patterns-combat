"""Feature engineering layer."""

from .encoder import EncodedBatch, encode_features
from .engineering import (
    FeatureEngineer,
    FeatureEngineeringConfig,
    FeatureEngineeringResult,
    FeatureValidationReport,
    export_feature_sets,
)
from .hidden_state_features import (
    HIDDEN_STATE_FEATURE_COLUMNS,
    HiddenStateFeatureLayer,
    build_hidden_state_feature_layer,
)

__all__ = [
    "EncodedBatch",
    "encode_features",
    "FeatureEngineer",
    "FeatureEngineeringConfig",
    "FeatureEngineeringResult",
    "FeatureValidationReport",
    "export_feature_sets",
    "HiddenStateFeatureLayer",
    "HIDDEN_STATE_FEATURE_COLUMNS",
    "build_hidden_state_feature_layer",
]
