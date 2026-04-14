from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    log_likelihood: float
    transition_matrix: np.ndarray
    converged: bool
    n_iterations: int
    last_delta: float | None


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
        hmm_logger = logging.getLogger("hmmlearn")
        prev_level = hmm_logger.level
        hmm_logger.setLevel(logging.ERROR)
        try:
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
        finally:
            hmm_logger.setLevel(prev_level)

        monitor = getattr(self.model, "monitor_", None)
        history = list(getattr(monitor, "history", [])) if monitor is not None else []
        last_delta = None
        if len(history) >= 2:
            last_delta = float(history[-1] - history[-2])
            if last_delta < 0:
                logger.warning(
                    "HMM finished with negative monitor delta: delta=%.6f (possible local instability).",
                    last_delta,
                )
        converged = bool(getattr(monitor, "converged", True))
        n_iterations = int(len(history))

        return TrainingResult(
            log_likelihood=score,
            transition_matrix=self.model.transmat_.copy(),
            converged=converged,
            n_iterations=n_iterations,
            last_delta=last_delta,
        )

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
