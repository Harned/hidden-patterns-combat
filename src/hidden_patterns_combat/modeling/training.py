from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingResult:
    log_likelihood: float
    transition_matrix: np.ndarray


class HMMTrainer:
    def __init__(self, model, topology_mode: str = "ergodic"):
        self.model = model
        self.topology_mode = topology_mode

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
            self._apply_topology_constraints()
            score = float(self.model.score(x, lengths=lengths))
        else:
            self.model.fit(x)
            self.model.transmat_ = self._repair_transmat(self.model.transmat_)
            self._apply_topology_constraints()
            score = float(self.model.score(x))
        return TrainingResult(log_likelihood=score, transition_matrix=self.model.transmat_.copy())

    def _apply_topology_constraints(self) -> None:
        if self.topology_mode != "left_to_right":
            return

        n = self.model.transmat_.shape[0]
        mask = np.zeros((n, n), dtype=float)
        for i in range(n):
            mask[i, i] = 1.0
            if i + 1 < n:
                mask[i, i + 1] = 1.0

        constrained = self.model.transmat_ * mask
        for i in range(n):
            row_sum = constrained[i].sum()
            if row_sum <= 0:
                constrained[i, i] = 1.0
                if i + 1 < n:
                    constrained[i, i + 1] = 1.0
                row_sum = constrained[i].sum()
            constrained[i] = constrained[i] / row_sum

        self.model.transmat_ = constrained
        startprob = np.zeros(n, dtype=float)
        startprob[0] = 1.0
        self.model.startprob_ = startprob
