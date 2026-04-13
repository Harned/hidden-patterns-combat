from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DecodingResult:
    states: np.ndarray
    state_probabilities: np.ndarray
    log_likelihood: float


class HMMDecoder:
    def __init__(self, model):
        self.model = model

    def decode(self, x: np.ndarray, lengths: list[int] | None = None) -> DecodingResult:
        if lengths:
            states = self.model.predict(x, lengths=lengths)
            probs = self.model.predict_proba(x, lengths=lengths)
            score = float(self.model.score(x, lengths=lengths))
        else:
            states = self.model.predict(x)
            probs = self.model.predict_proba(x)
            score = float(self.model.score(x))
        return DecodingResult(states=states, state_probabilities=probs, log_likelihood=score)
