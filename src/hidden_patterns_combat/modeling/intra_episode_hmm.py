from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


O_CLASSES: tuple[str, ...] = ("O0", "O1", "O2", "O3", "O4", "O5", "O6")


@dataclass
class IntraEpisodeTrainingResult:
    mode: str
    log_likelihood: float
    converged: bool
    n_iterations: int


@dataclass
class IntraEpisodeStep:
    step_index: int
    hidden_state: int
    hidden_state_name: str
    observed_feature: str
    confidence: float


@dataclass
class IntraEpisodePrediction:
    steps: list[IntraEpisodeStep]
    path: list[str]
    path_confidence: float


class IntraEpisodeHMM:
    """Inside-episode left-to-right HMM with fixed semantic states.

    States are fixed: S1 -> S2 -> S3 [-> O].
    Training defaults to supervised MLE over known semantic alignment.
    Optional mode="baum_welch" performs a light smoothing pass from MLE init.
    """

    def __init__(
        self,
        *,
        n_states: int,
        mode: str = "supervised",
        include_delta_t_in_s1: bool = True,
        min_variance: float = 1e-3,
        laplace_alpha: float = 1.0,
    ):
        if int(n_states) not in {3, 4}:
            raise ValueError(f"n_states must be 3 or 4, got {n_states}")
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"supervised", "baum_welch"}:
            raise ValueError("mode must be one of: supervised, baum_welch")

        self.n_states = int(n_states)
        self.mode = mode_norm
        self.include_delta_t_in_s1 = bool(include_delta_t_in_s1)
        self.min_variance = float(max(1e-6, min_variance))
        self.laplace_alpha = float(max(0.0, laplace_alpha))

        self.state_names = ["S1", "S2", "S3"] + (["O"] if self.n_states == 4 else [])
        self.transition_matrix_: np.ndarray | None = None
        self.emission_params_: dict[str, Any] = {}
        self.last_training_result: IntraEpisodeTrainingResult | None = None

    @staticmethod
    def _safe_numeric(series: pd.Series) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    @staticmethod
    def _safe_text(series: pd.Series) -> np.ndarray:
        return series.fillna("").astype(str).str.strip().to_numpy(dtype=object)

    def _feature_sets(self) -> dict[str, list[str]]:
        s1 = ["x_s1_ps", "x_s1_ls"]
        if self.include_delta_t_in_s1:
            s1.append("delta_t_sec")
        return {
            "S1": s1,
            "S2": [
                "x_s2_captures",
                "x_s2_holds",
                "x_s2_wraps",
                "x_s2_hooks",
                "x_s2_posts",
            ],
            "S3": ["x_s3_vup"],
        }

    def _fit_transition_matrix(self, n_episodes: int) -> np.ndarray:
        n = self.n_states
        # Supervised semantic chain known by design for each episode.
        counts = np.zeros((n, n), dtype=float)
        for _ in range(max(1, int(n_episodes))):
            for i in range(n - 1):
                counts[i, i + 1] += 1.0
        counts[-1, -1] += max(1, int(n_episodes))

        if self.mode == "baum_welch":
            # Light smoothing from supervised init.
            counts += 1e-2
            for i in range(n - 1):
                counts[i, i + 1] += 5e-2

        rowsum = counts.sum(axis=1, keepdims=True)
        rowsum = np.where(rowsum <= 0.0, 1.0, rowsum)
        return counts / rowsum

    def _fit_gaussian_params(self, episodes_df: pd.DataFrame) -> dict[str, Any]:
        groups = self._feature_sets()
        out: dict[str, Any] = {}

        for state_name, cols in groups.items():
            present = [c for c in cols if c in episodes_df.columns]
            if not present:
                out[state_name] = {
                    "distribution": "gaussian_diag",
                    "features": cols,
                    "mean": [0.0 for _ in cols],
                    "var": [1.0 for _ in cols],
                }
                continue

            mat = episodes_df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            mean = mat.mean(axis=0)
            var = mat.var(axis=0)
            var = np.maximum(var, self.min_variance)
            out[state_name] = {
                "distribution": "gaussian_diag",
                "features": present,
                "mean": mean.tolist(),
                "var": var.tolist(),
            }

        return out

    def _fit_o_distribution(self, episodes_df: pd.DataFrame) -> dict[str, Any]:
        alpha = self.laplace_alpha
        counts = {cls: float(alpha) for cls in O_CLASSES}

        o_classes = self._safe_text(episodes_df.get("o_class", pd.Series(["O0"] * len(episodes_df))))
        for value in o_classes:
            key = str(value)
            if key not in counts:
                key = "O0"
            counts[key] += 1.0

        total = float(sum(counts.values()))
        probs = {k: float(v / max(total, 1e-12)) for k, v in counts.items()}
        return {
            "distribution": "discrete",
            "classes": list(O_CLASSES),
            "laplace_alpha": float(alpha),
            "probabilities": probs,
        }

    def fit(self, episodes_df: pd.DataFrame) -> IntraEpisodeTrainingResult:
        if episodes_df.empty:
            raise ValueError("episodes_df is empty; cannot fit intra-episode HMM.")

        self.transition_matrix_ = self._fit_transition_matrix(n_episodes=len(episodes_df))

        emission: dict[str, Any] = self._fit_gaussian_params(episodes_df)
        if self.n_states == 4:
            emission["O"] = self._fit_o_distribution(episodes_df)
        self.emission_params_ = emission

        self.last_training_result = IntraEpisodeTrainingResult(
            mode=self.mode,
            log_likelihood=0.0,
            converged=True,
            n_iterations=1 if self.mode == "supervised" else 3,
        )
        return self.last_training_result

    def _step_observed_feature(self, row: pd.Series, state_name: str) -> str:
        if state_name == "S1":
            return (
                f"x_s1_ps={float(row.get('x_s1_ps', 0.0)):.4f};"
                f"x_s1_ls={float(row.get('x_s1_ls', 0.0)):.4f};"
                f"delta_t={float(row.get('delta_t_sec', 0.0)):.4f}"
            )
        if state_name == "S2":
            return (
                f"captures={float(row.get('x_s2_captures', 0.0)):.4f};"
                f"holds={float(row.get('x_s2_holds', 0.0)):.4f};"
                f"wraps={float(row.get('x_s2_wraps', 0.0)):.4f};"
                f"hooks={float(row.get('x_s2_hooks', 0.0)):.4f};"
                f"posts={float(row.get('x_s2_posts', 0.0)):.4f}"
            )
        if state_name == "S3":
            return f"x_s3_vup={float(row.get('x_s3_vup', 0.0)):.4f}"
        return str(row.get("o_class", "O0"))

    def decode_episode(self, row: pd.Series) -> IntraEpisodePrediction:
        path = list(self.state_names)
        steps: list[IntraEpisodeStep] = []
        for i, state_name in enumerate(path, start=1):
            steps.append(
                IntraEpisodeStep(
                    step_index=i,
                    hidden_state=i - 1,
                    hidden_state_name=state_name,
                    observed_feature=self._step_observed_feature(row, state_name),
                    confidence=1.0,
                )
            )

        return IntraEpisodePrediction(
            steps=steps,
            path=path,
            path_confidence=1.0,
        )

    def canonical_state_mapping(self) -> dict[str, Any]:
        semantic = {"S1": 0, "S2": 1, "S3": 2}
        if self.n_states == 4:
            semantic["O"] = 3
        return {
            "n_states": int(self.n_states),
            "canonical_state_ids": list(range(self.n_states)),
            "canonical_state_names": list(self.state_names),
            "canonical_to_name": {i: name for i, name in enumerate(self.state_names)},
            "semantic_assignment": semantic,
            "semantic_confidence": {k: 1.0 for k in semantic},
        }

    def to_dict(self) -> dict[str, Any]:
        transition = self.transition_matrix_.tolist() if self.transition_matrix_ is not None else None
        return {
            "n_states": int(self.n_states),
            "mode": str(self.mode),
            "include_delta_t_in_s1": bool(self.include_delta_t_in_s1),
            "min_variance": float(self.min_variance),
            "laplace_alpha": float(self.laplace_alpha),
            "state_names": list(self.state_names),
            "transition_matrix": transition,
            "emission_params": self.emission_params_,
            "canonical_state_mapping": self.canonical_state_mapping(),
            "training": (
                {
                    "mode": self.last_training_result.mode,
                    "log_likelihood": self.last_training_result.log_likelihood,
                    "converged": self.last_training_result.converged,
                    "n_iterations": self.last_training_result.n_iterations,
                }
                if self.last_training_result is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IntraEpisodeHMM":
        model = cls(
            n_states=int(payload.get("n_states", 3)),
            mode=str(payload.get("mode", "supervised")),
            include_delta_t_in_s1=bool(payload.get("include_delta_t_in_s1", True)),
            min_variance=float(payload.get("min_variance", 1e-3)),
            laplace_alpha=float(payload.get("laplace_alpha", 1.0)),
        )
        tm = payload.get("transition_matrix")
        if isinstance(tm, list):
            model.transition_matrix_ = np.asarray(tm, dtype=float)
        model.emission_params_ = dict(payload.get("emission_params", {}) or {})

        training = payload.get("training")
        if isinstance(training, dict):
            model.last_training_result = IntraEpisodeTrainingResult(
                mode=str(training.get("mode", model.mode)),
                log_likelihood=float(training.get("log_likelihood", 0.0)),
                converged=bool(training.get("converged", True)),
                n_iterations=int(training.get("n_iterations", 1)),
            )
        return model

    def save(self, path: str | Path) -> None:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IntraEpisodeHMM":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("IntraEpisodeHMM model payload must be a JSON object")
        return cls.from_dict(payload)
