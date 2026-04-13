from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingResult:
    log_likelihood: float
    transition_matrix: np.ndarray


class HMMTrainer:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _repair_transmat(transmat: np.ndarray) -> np.ndarray:
        fixed = transmat.copy()
        n = fixed.shape[1]
        for i in range(fixed.shape[0]):
            s = fixed[i].sum()
            if s <= 0:
                fixed[i] = np.ones(n) / n
            else:
                fixed[i] = fixed[i] / s
        return fixed

    def fit(self, x: np.ndarray, lengths: list[int] | None = None) -> TrainingResult:
        if lengths:
            self.model.fit(x, lengths=lengths)
            self.model.transmat_ = self._repair_transmat(self.model.transmat_)
            score = float(self.model.score(x, lengths=lengths))
        else:
            self.model.fit(x)
            self.model.transmat_ = self._repair_transmat(self.model.transmat_)
            score = float(self.model.score(x))
        return TrainingResult(log_likelihood=score, transition_matrix=self.model.transmat_.copy())
