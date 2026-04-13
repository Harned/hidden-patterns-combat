from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hidden_patterns_combat.config import ModelConfig

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


class HMMEngine:
    def __init__(self, cfg: ModelConfig):
        if GaussianHMM is None:
            raise ImportError(
                "hmmlearn is required for modeling. Install dependencies with `pip install -e .`"
            ) from IMPORT_ERROR

        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=cfg.n_hidden_states,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_iter,
            random_state=cfg.random_state,
        )

    def fit(self, features: pd.DataFrame) -> float:
        x = self.scaler.fit_transform(features.to_numpy(dtype=float))
        self.model.fit(x)
        score = float(self.model.score(x))
        logger.info("HMM fitted. log_likelihood=%.4f", score)
        return score

    def predict(self, features: pd.DataFrame) -> HMMPrediction:
        x = self.scaler.transform(features.to_numpy(dtype=float))
        states = self.model.predict(x)
        probs = self.model.predict_proba(x)
        score = float(self.model.score(x))
        return HMMPrediction(states=states, state_probabilities=probs, log_likelihood=score)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"cfg": self.cfg, "scaler": self.scaler, "model": self.model}, f)

    @classmethod
    def load(cls, path: str | Path) -> "HMMEngine":
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        obj = cls(payload["cfg"])
        obj.scaler = payload["scaler"]
        obj.model = payload["model"]
        return obj
