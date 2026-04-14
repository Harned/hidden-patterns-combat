from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hidden_patterns_combat.config import ModelConfig
from hidden_patterns_combat.modeling.decoding import HMMDecoder
from hidden_patterns_combat.modeling.observation_encoding import encode_observations
from hidden_patterns_combat.modeling.state_definition import StateDefinition, build_semantic_state_definition
from hidden_patterns_combat.modeling.training import HMMTrainer

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
except Exception as exc:  # pragma: no cover
    GaussianHMM = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@dataclass
class HMMPrediction:
    states: np.ndarray
    state_probabilities: np.ndarray
    log_likelihood: float
    state_names: list[str]


class HMMEngine:
    """HMM modeling facade with explicit state/observation/training/decoding layers."""

    def __init__(self, cfg: ModelConfig, state_definition: StateDefinition | None = None):
        if GaussianHMM is None:
            raise ImportError(
                "hmmlearn is required for modeling. Install dependencies with `pip install -e .`"
            ) from IMPORT_ERROR

        self.cfg = cfg
        self.state_definition = state_definition or StateDefinition.research_default(cfg.n_hidden_states)
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=cfg.n_hidden_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            random_state=cfg.random_state,
        )
        self.feature_columns: list[str] | None = None

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            return features
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Missing required HMM feature columns: {missing}")
        return features[self.feature_columns]

    def fit(self, features: pd.DataFrame, sequence_ids: pd.Series | None = None) -> float:
        features = features.copy().reset_index(drop=True)
        batch = encode_observations(
            features=features,
            scaler=self.scaler,
            fit_scaler=True,
            sequence_ids=sequence_ids,
        )
        self.feature_columns = batch.feature_columns
        trainer = HMMTrainer(self.model, topology_mode=self.cfg.topology_mode)
        result = trainer.fit(batch.values, lengths=batch.lengths)
        train_states = self.model.predict(batch.values, lengths=batch.lengths if batch.lengths else None)
        self.state_definition = build_semantic_state_definition(
            features=features,
            decoded_states=train_states,
            n_states=self.cfg.n_hidden_states,
        )
        logger.info(
            "HMM fitted. log_likelihood=%.4f, n_sequences=%d",
            result.log_likelihood,
            len(batch.lengths) if batch.lengths else 1,
        )
        return result.log_likelihood

    def predict(self, features: pd.DataFrame, sequence_ids: pd.Series | None = None) -> HMMPrediction:
        features = self._align_features(features)
        batch = encode_observations(
            features=features,
            scaler=self.scaler,
            fit_scaler=False,
            sequence_ids=sequence_ids,
        )
        decoder = HMMDecoder(self.model)
        result = decoder.decode(batch.values, lengths=batch.lengths)
        return HMMPrediction(
            states=result.states,
            state_probabilities=result.state_probabilities,
            log_likelihood=result.log_likelihood,
            state_names=[self.state_definition.state_name(int(s)) for s in result.states],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "scaler": self.scaler,
            "model": self.model,
            "state_definition": self.state_definition.to_dict(),
            "feature_columns": self.feature_columns,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "HMMEngine":
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        state_definition = StateDefinition.from_dict(payload.get("state_definition", {}))
        if not state_definition.states:
            state_definition = StateDefinition.research_default(payload["cfg"].n_hidden_states)

        obj = cls(payload["cfg"], state_definition=state_definition)
        obj.scaler = payload["scaler"]
        obj.model = payload["model"]
        obj.feature_columns = payload.get("feature_columns")
        return obj
