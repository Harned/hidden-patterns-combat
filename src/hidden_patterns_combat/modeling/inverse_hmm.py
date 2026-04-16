from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from hidden_patterns_combat.config import ModelConfig
from hidden_patterns_combat.modeling.observation_encoding import build_lengths
from hidden_patterns_combat.modeling.state_definition import SemanticOrderingDiagnostics, StateDefinition

logger = logging.getLogger(__name__)

_FINISH_LIKE_OBSERVED_CLASSES = {"zap_t", "hold", "arm_submission", "leg_submission"}
_SEMANTIC_NAMES = ("S1", "S2", "S3")
_EPS = 1e-12


@dataclass
class InverseTrainingResult:
    log_likelihood: float
    converged: bool
    n_iterations: int
    last_delta: float | None


@dataclass
class InverseHMMPrediction:
    states: np.ndarray
    state_probabilities: np.ndarray
    log_likelihood: float
    state_names: list[str]


class InverseDiagnosticHMM:
    """Discrete HMM for inverse diagnostic mode.

    Observed layer stays in observed ZAP classes. Hidden-state identifiability is
    stabilized by weak supervision anchors and constrained left-to-right topology.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        observation_classes: list[str],
        state_definition: StateDefinition | None = None,
    ):
        self.cfg = cfg
        self.n_states = int(cfg.n_hidden_states)
        self.observation_classes = list(observation_classes)
        self.obs_to_idx = {name: i for i, name in enumerate(self.observation_classes)}
        self.idx_to_obs = {i: name for name, i in self.obs_to_idx.items()}
        self.n_observations = len(self.observation_classes)

        self.state_definition = state_definition or StateDefinition.research_default(self.n_states)

        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.emissionprob_: np.ndarray | None = None

        self.last_training_result: InverseTrainingResult | None = None
        self.last_semantic_diagnostics: dict[str, object] | None = None
        self.last_canonical_state_mapping: dict[str, object] | None = None

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        total = float(v.sum())
        if total <= 0.0:
            return np.ones_like(v) / max(1, len(v))
        return v / total

    @staticmethod
    def _normalize_rows(m: np.ndarray) -> np.ndarray:
        out = m.copy().astype(float)
        for i in range(out.shape[0]):
            row_sum = float(out[i].sum())
            if row_sum <= 0.0:
                out[i] = np.ones(out.shape[1], dtype=float) / float(out.shape[1])
            else:
                out[i] = out[i] / row_sum
        return out

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        out = values.astype(float).copy()
        if out.size == 0:
            return out
        vmin = float(np.nanmin(out))
        vmax = float(np.nanmax(out))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            return np.ones_like(out, dtype=float) * 0.5
        return (out - vmin) / (vmax - vmin)

    @staticmethod
    def _quantile_bin(values: pd.Series, q_low: float = 0.33, q_high: float = 0.66) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
        positive = numeric[numeric > 0]
        if positive.empty:
            return pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
        lo = float(positive.quantile(q_low))
        hi = float(positive.quantile(q_high))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = lo
        if hi <= lo:
            hi = lo + 1e-6
        out = pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
        out = out.where(numeric <= lo, 1.0)
        out = out.where(numeric <= hi, 2.0)
        return out

    def _init_startprob_transmat(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_states
        if self.cfg.topology_mode == "left_to_right":
            start = np.zeros(n, dtype=float)
            start[0] = 1.0

            trans = np.zeros((n, n), dtype=float)
            init_self = float(getattr(self.cfg, "inverse_initial_self_transition", 0.88))
            init_self = max(0.55, min(0.98, init_self))
            init_forward = 1.0 - init_self
            for i in range(n):
                if i + 1 < n:
                    trans[i, i] = init_self
                    trans[i, i + 1] = init_forward
                else:
                    trans[i, i] = 1.0
            trans[-1, :] = 0.0
            trans[-1, -1] = 1.0
            return start, self._normalize_rows(trans)

        start = np.ones(n, dtype=float) / float(n)
        trans = np.ones((n, n), dtype=float)
        return start, self._normalize_rows(trans)

    def _semantic_emission_prior(self) -> np.ndarray:
        n = self.n_states
        m = self.n_observations
        emissions = np.ones((n, m), dtype=float)
        class_idx = self.obs_to_idx

        def bump(state: int, class_name: str, weight: float) -> None:
            idx = class_idx.get(class_name)
            if idx is not None:
                emissions[state, idx] += float(weight)

        if n >= 1:
            bump(0, "zap_r", 4.0)
            bump(0, "zap_n", 1.5)
            bump(0, "no_score", 1.0)

        if n >= 2:
            bump(1, "zap_n", 4.0)
            bump(1, "hold", 2.2)
            bump(1, "zap_r", 1.2)
            bump(1, "no_score", 0.6)

        if n >= 3:
            bump(2, "zap_t", 4.2)
            bump(2, "arm_submission", 4.0)
            bump(2, "leg_submission", 4.0)
            bump(2, "hold", 2.0)
            bump(2, "zap_n", 1.2)
            bump(2, "no_score", 0.2)

        for state_id in range(n):
            bump(state_id, "unknown", 0.15)

        return self._normalize_rows(emissions)

    def _initialize_parameters(self) -> None:
        start, trans = self._init_startprob_transmat()
        emissions = self._semantic_emission_prior()
        self.startprob_ = start
        self.transmat_ = trans
        self.emissionprob_ = emissions

    def _encode_observations(self, observed: pd.Series | list[str]) -> np.ndarray:
        if isinstance(observed, pd.Series):
            values = observed.fillna("unknown").astype(str).tolist()
        else:
            values = ["unknown" if v is None else str(v) for v in observed]
        unknown_idx = self.obs_to_idx.get("unknown", 0)
        return np.array([self.obs_to_idx.get(v, unknown_idx) for v in values], dtype=int)

    def _sequence_lengths(self, sequence_ids: pd.Series | None) -> list[int]:
        return build_lengths(sequence_ids)

    @staticmethod
    def _split_by_lengths(values: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
        if not lengths:
            return [values]
        if int(sum(lengths)) != int(len(values)):
            raise ValueError(
                "Invalid sequence lengths for inverse HMM: "
                f"sum(lengths)={sum(lengths)} values={len(values)}"
            )
        out: list[np.ndarray] = []
        start = 0
        for length in lengths:
            end = start + int(length)
            out.append(values[start:end])
            start = end
        return out

    @staticmethod
    def _state_prior_multiplier(
        state_prior: np.ndarray | None,
        t: int,
        n_states: int,
        *,
        strength: float,
    ) -> np.ndarray:
        if state_prior is None or strength <= 0.0:
            return np.ones(n_states, dtype=float)
        if t < 0 or t >= len(state_prior):
            return np.ones(n_states, dtype=float)
        row = np.asarray(state_prior[t], dtype=float).reshape(-1)
        if row.size != n_states:
            return np.ones(n_states, dtype=float)
        row = np.clip(row, _EPS, None)
        row /= max(float(row.sum()), _EPS)
        return np.power(row, float(np.clip(strength, 0.0, 2.0)))

    def _forward_backward(
        self,
        obs: np.ndarray,
        *,
        state_prior: np.ndarray | None = None,
        state_prior_strength: float = 0.0,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.emissionprob_ is not None

        n = self.n_states
        t_max = len(obs)
        eps = _EPS

        alpha = np.zeros((t_max, n), dtype=float)
        beta = np.zeros((t_max, n), dtype=float)
        scales = np.zeros(t_max, dtype=float)

        emit0 = self.emissionprob_[:, obs[0]] * self._state_prior_multiplier(
            state_prior,
            0,
            n,
            strength=state_prior_strength,
        )
        alpha[0] = self.startprob_ * emit0
        scales[0] = max(float(alpha[0].sum()), eps)
        alpha[0] /= scales[0]

        for t in range(1, t_max):
            emit_t = self.emissionprob_[:, obs[t]] * self._state_prior_multiplier(
                state_prior,
                t,
                n,
                strength=state_prior_strength,
            )
            alpha[t] = (alpha[t - 1] @ self.transmat_) * emit_t
            scales[t] = max(float(alpha[t].sum()), eps)
            alpha[t] /= scales[t]

        beta[-1] = np.ones(n, dtype=float)
        for t in range(t_max - 2, -1, -1):
            emit_next = self.emissionprob_[:, obs[t + 1]] * self._state_prior_multiplier(
                state_prior,
                t + 1,
                n,
                strength=state_prior_strength,
            )
            beta[t] = (self.transmat_ * emit_next.reshape(1, -1)) @ beta[t + 1]
            beta[t] /= max(scales[t + 1], eps)

        gamma = alpha * beta
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), eps)

        xi_sum = np.zeros((n, n), dtype=float)
        xi_steps = np.zeros((max(0, t_max - 1), n, n), dtype=float)
        for t in range(t_max - 1):
            numer = (
                alpha[t].reshape(-1, 1)
                * self.transmat_
                * (
                    self.emissionprob_[:, obs[t + 1]]
                    * self._state_prior_multiplier(
                        state_prior,
                        t + 1,
                        n,
                        strength=state_prior_strength,
                    )
                ).reshape(1, -1)
                * beta[t + 1].reshape(1, -1)
            )
            denom = max(float(numer.sum()), eps)
            xi_t = numer / denom
            xi_steps[t] = xi_t
            xi_sum += xi_t

        log_likelihood = float(np.sum(np.log(np.maximum(scales, eps))))
        return log_likelihood, gamma, xi_sum, xi_steps

    def _viterbi(
        self,
        obs: np.ndarray,
        *,
        state_prior: np.ndarray | None = None,
        state_prior_strength: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.emissionprob_ is not None

        eps = 1e-12
        log_start = np.log(np.maximum(self.startprob_, eps))
        log_trans = np.log(np.maximum(self.transmat_, eps))
        log_emit = np.log(np.maximum(self.emissionprob_, eps))

        t_max = len(obs)
        n = self.n_states
        delta = np.zeros((t_max, n), dtype=float)
        psi = np.zeros((t_max, n), dtype=int)
        prior0 = np.log(
            np.maximum(
                self._state_prior_multiplier(
                    state_prior,
                    0,
                    n,
                    strength=state_prior_strength,
                ),
                eps,
            )
        )
        delta[0] = log_start + log_emit[:, obs[0]] + prior0

        for t in range(1, t_max):
            scores = delta[t - 1].reshape(-1, 1) + log_trans
            psi[t] = np.argmax(scores, axis=0)
            prior_t = np.log(
                np.maximum(
                    self._state_prior_multiplier(
                        state_prior,
                        t,
                        n,
                        strength=state_prior_strength,
                    ),
                    eps,
                )
            )
            delta[t] = np.max(scores, axis=0) + log_emit[:, obs[t]] + prior_t

        states = np.zeros(t_max, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(t_max - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])
        return states, float(np.max(delta[-1]))

    def _enforce_topology(self, trans: np.ndarray) -> np.ndarray:
        out = trans.copy().astype(float)
        if self.cfg.topology_mode != "left_to_right":
            return self._normalize_rows(out)

        n = out.shape[0]
        min_forward = float(getattr(self.cfg, "min_forward_transition", 0.05))
        max_self = float(getattr(self.cfg, "max_self_transition", 0.94))
        min_self = float(getattr(self.cfg, "min_self_transition", 0.55))
        first_min_forward = float(getattr(self.cfg, "inverse_first_state_min_forward_transition", 0.15))
        first_max_self = float(getattr(self.cfg, "inverse_first_state_max_self_transition", 0.85))
        min_forward = max(0.0, min(0.49, min_forward))
        max_self = max(0.50, min(0.995, max_self))
        min_self = max(0.05, min(max_self - 1e-6, min_self))
        first_min_forward = max(min_forward, min(0.49, first_min_forward))
        first_max_self = max(0.50, min(max_self, first_max_self))
        if min_self + min_forward >= 1.0:
            min_self = max(0.05, 1.0 - min_forward - 1e-6)

        mask = np.zeros((n, n), dtype=float)
        for i in range(n):
            mask[i, i] = 1.0
            if i + 1 < n:
                mask[i, i + 1] = 1.0
        constrained = out * mask

        for i in range(n):
            if i == n - 1:
                constrained[i, :] = 0.0
                constrained[i, i] = 1.0
                continue
            row_min_forward = first_min_forward if i == 0 else min_forward
            row_max_self = first_max_self if i == 0 else max_self
            row_min_self = min_self
            if row_min_self + row_min_forward >= 1.0:
                row_min_self = max(0.05, 1.0 - row_min_forward - 1e-6)
            row_min_self = min(row_min_self, row_max_self - 1e-6)

            self_prob = max(float(constrained[i, i]), _EPS)
            next_prob = max(float(constrained[i, i + 1]), _EPS)

            row_sum = self_prob + next_prob
            self_prob /= row_sum
            next_prob /= row_sum

            if self_prob > row_max_self:
                spill = self_prob - row_max_self
                self_prob = row_max_self
                next_prob += spill

            if self_prob < row_min_self:
                need = row_min_self - self_prob
                take = min(need, next_prob - _EPS)
                self_prob += take
                next_prob -= take

            if next_prob < row_min_forward:
                need = row_min_forward - next_prob
                take = min(need, self_prob - _EPS)
                next_prob += take
                self_prob -= take

            norm = max(self_prob + next_prob, _EPS)
            constrained[i, :] = 0.0
            constrained[i, i] = self_prob / norm
            constrained[i, i + 1] = next_prob / norm

        return constrained

    def _canonical_semantic_payload(self, semantic_diag: SemanticOrderingDiagnostics) -> dict[str, object]:
        canonical_order = [int(x) for x in semantic_diag.canonical_order]
        inverse = np.zeros(len(canonical_order), dtype=int)
        for new_idx, old_idx in enumerate(canonical_order):
            inverse[int(old_idx)] = int(new_idx)

        semantic_to_canonical: dict[str, int] = {}
        for semantic_name, old_state in semantic_diag.semantic_to_original_state.items():
            semantic_to_canonical[semantic_name] = int(inverse[int(old_state)])

        profiles: list[dict[str, object]] = []
        for row in semantic_diag.state_profiles:
            old_state = int(row.get("state_id", -1))
            profiles.append(
                {
                    **row,
                    "original_state_id": old_state,
                    "state_id": int(inverse[old_state]) if old_state >= 0 else -1,
                }
            )
        profiles = sorted(profiles, key=lambda item: int(item.get("state_id", 10**6)))

        return {
            "canonical_order_from_original": canonical_order,
            "semantic_to_state": semantic_to_canonical,
            "semantic_confidence": {k: float(v) for k, v in semantic_diag.semantic_confidence.items()},
            "state_profiles": profiles,
            "warnings": list(semantic_diag.warnings),
            "semantic_order_matches_topology_before_reorder": bool(
                semantic_diag.semantic_order_matches_topology
            ),
        }

    def _build_canonical_state_mapping(self) -> dict[str, object]:
        n_states = self.n_states
        canonical_ids = list(range(n_states))
        canonical_to_name = {idx: self.state_definition.state_name(idx) for idx in canonical_ids}
        canonical_names = [canonical_to_name[idx] for idx in canonical_ids]

        diagnostics = self.last_semantic_diagnostics or {}
        canonical_from_original = diagnostics.get("canonical_order_from_original", canonical_ids)
        if not isinstance(canonical_from_original, list) or len(canonical_from_original) != n_states:
            canonical_from_original = canonical_ids

        canonical_to_original = {new: int(old) for new, old in enumerate(canonical_from_original)}
        original_to_canonical = {int(old): new for new, old in canonical_to_original.items()}

        return {
            "n_states": n_states,
            "canonical_state_ids": canonical_ids,
            "canonical_state_names": canonical_names,
            "canonical_to_name": {int(k): str(v) for k, v in canonical_to_name.items()},
            "canonical_to_original": canonical_to_original,
            "original_to_canonical": original_to_canonical,
            "semantic_assignment": {
                str(k): int(v) for k, v in (diagnostics.get("semantic_to_state", {}) or {}).items()
            },
            "semantic_confidence": {
                str(k): float(v) for k, v in (diagnostics.get("semantic_confidence", {}) or {}).items()
            },
            "semantic_order_matches_topology_before_reorder": diagnostics.get(
                "semantic_order_matches_topology_before_reorder"
            ),
        }

    def _reorder_states(self, order: list[int]) -> None:
        if self.startprob_ is None or self.transmat_ is None or self.emissionprob_ is None:
            return
        if sorted(order) != list(range(self.n_states)):
            raise ValueError(f"Invalid state order for reordering: {order}")

        perm = np.asarray(order, dtype=int)
        self.startprob_ = self.startprob_[perm]
        self.transmat_ = self.transmat_[perm][:, perm]
        self.emissionprob_ = self.emissionprob_[perm]
        if self.cfg.topology_mode == "left_to_right":
            self.transmat_ = self._enforce_topology(self.transmat_)
            self.startprob_ = np.zeros(self.n_states, dtype=float)
            self.startprob_[0] = 1.0

    @staticmethod
    def _safe_num_col(frame: pd.DataFrame, name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).astype(float)

    @staticmethod
    def _safe_text_col(frame: pd.DataFrame, name: str, default: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series([default] * len(frame), index=frame.index, dtype="object")
        return (
            frame[name]
            .fillna(default)
            .astype(str)
            .str.strip()
            .replace({"": default, "nan": default, "None": default})
        )

    def _semantic_feature_view(self, hidden_features: pd.DataFrame) -> pd.DataFrame:
        frame = hidden_features.copy().reset_index(drop=True)
        out = pd.DataFrame(index=frame.index)

        out["maneuver_right_code"] = self._safe_num_col(frame, "maneuver_right_code")
        out["maneuver_left_code"] = self._safe_num_col(frame, "maneuver_left_code")
        out["grips_code"] = self._safe_num_col(frame, "kfv_capture_code")
        out["holds_code"] = self._safe_num_col(frame, "kfv_grip_code")
        out["bodylocks_code"] = self._safe_num_col(frame, "kfv_wrap_code")
        out["underhooks_code"] = self._safe_num_col(frame, "kfv_hook_code")
        out["posts_code"] = self._safe_num_col(frame, "kfv_post_code")
        out["kfv_code"] = (
            out["grips_code"]
            + out["holds_code"]
            + out["bodylocks_code"]
            + out["underhooks_code"]
            + out["posts_code"]
        )
        out["vup_code"] = self._safe_num_col(frame, "vup_code")
        out["duration"] = self._safe_num_col(frame, "episode_time_sec")
        out["pause"] = self._safe_num_col(frame, "pause_time_sec")

        duration_bin = self._safe_num_col(frame, "duration_bin")
        if float(duration_bin.abs().sum()) <= 0.0:
            duration_bin = self._quantile_bin(out["duration"])
        pause_bin = self._safe_num_col(frame, "pause_bin")
        if float(pause_bin.abs().sum()) <= 0.0:
            pause_bin = self._quantile_bin(out["pause"])
        out["duration_bin"] = duration_bin.astype(float)
        out["pause_bin"] = pause_bin.astype(float)
        out["sequence_progress"] = self._safe_num_col(frame, "sequence_progress").clip(lower=0.0, upper=1.0)

        anchor_s1 = self._safe_num_col(frame, "anchor_s1")
        anchor_s2 = self._safe_num_col(frame, "anchor_s2")
        anchor_s3 = self._safe_num_col(frame, "anchor_s3")

        if float(anchor_s1.abs().sum() + anchor_s2.abs().sum() + anchor_s3.abs().sum()) <= 0.0:
            dur_norm = out["duration_bin"] / max(1.0, float(out["duration_bin"].max()))
            pause_norm = out["pause_bin"] / max(1.0, float(out["pause_bin"].max()))
            s1 = pd.Series(
                self._minmax((out["maneuver_right_code"] + out["maneuver_left_code"]).to_numpy()),
                index=out.index,
            )
            s2 = pd.Series(self._minmax(out["kfv_code"].to_numpy()), index=out.index)
            s3 = pd.Series(self._minmax(out["vup_code"].to_numpy()), index=out.index)
            anchor_frame = pd.DataFrame(
                {
                    "anchor_s1": (s1 + 0.15 * (1.0 - dur_norm)).clip(lower=0.0),
                    "anchor_s2": (s2 + 0.15 * dur_norm).clip(lower=0.0),
                    "anchor_s3": (s3 + 0.20 * dur_norm + 0.05 * (1.0 - pause_norm)).clip(lower=0.0),
                },
                index=out.index,
            )
            anchor_frame = anchor_frame + _EPS
            anchor_frame = anchor_frame.div(anchor_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
            anchor_s1 = anchor_frame["anchor_s1"]
            anchor_s2 = anchor_frame["anchor_s2"]
            anchor_s3 = anchor_frame["anchor_s3"]

        anchor_frame = pd.DataFrame(
            {
                "anchor_s1": anchor_s1.astype(float).clip(lower=0.0),
                "anchor_s2": anchor_s2.astype(float).clip(lower=0.0),
                "anchor_s3": anchor_s3.astype(float).clip(lower=0.0),
            },
            index=out.index,
        )
        anchor_frame = anchor_frame + _EPS
        anchor_frame = anchor_frame.div(anchor_frame.sum(axis=1).replace(0.0, 1.0), axis=0)
        out["anchor_s1"] = anchor_frame["anchor_s1"]
        out["anchor_s2"] = anchor_frame["anchor_s2"]
        out["anchor_s3"] = anchor_frame["anchor_s3"]
        out["train_weight"] = self._safe_num_col(frame, "train_weight").clip(lower=0.0, upper=1.0)
        out["observation_resolution_type"] = self._safe_text_col(frame, "observation_resolution_type", "unknown")
        out["observation_confidence_label"] = self._safe_text_col(frame, "observation_confidence_label", "low")

        return out

    def _row_train_weights(self, features: pd.DataFrame, observed_labels: list[str]) -> np.ndarray:
        weights = pd.to_numeric(features.get("train_weight"), errors="coerce").fillna(0.0).astype(float)
        if float(weights.abs().sum()) <= 0.0:
            weights = pd.Series(np.ones(len(features), dtype=float), index=features.index)
        observed = pd.Series(observed_labels, index=features.index).fillna("unknown").astype(str).str.lower()
        resolution = (
            pd.Series(features.get("observation_resolution_type", "unknown"), index=features.index)
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )
        confidence = (
            pd.Series(features.get("observation_confidence_label", "low"), index=features.index)
            .fillna("low")
            .astype(str)
            .str.lower()
        )

        observed_factor = observed.map({"no_score": 0.20, "unknown": 0.03}).fillna(1.0)
        resolution_factor = resolution.map(
            {
                "direct_finish_signal": 1.0,
                "inferred_from_score": 0.85,
                "no_score_rule": 0.20,
                "ambiguous": 0.02,
                "unknown": 0.01,
            }
        ).fillna(0.30)
        confidence_factor = confidence.map({"high": 1.0, "medium": 0.90, "low": 0.35}).fillna(0.35)
        weights = (weights * observed_factor * resolution_factor * confidence_factor).clip(lower=0.0, upper=1.0)
        return weights.to_numpy(dtype=float)

    def _row_emission_weights(self, features: pd.DataFrame, observed_labels: list[str]) -> np.ndarray:
        observed = pd.Series(observed_labels, index=features.index).fillna("unknown").astype(str).str.lower()
        resolution = (
            pd.Series(features.get("observation_resolution_type", "unknown"), index=features.index)
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )
        confidence = (
            pd.Series(features.get("observation_confidence_label", "low"), index=features.index)
            .fillna("low")
            .astype(str)
            .str.lower()
        )
        floor = float(getattr(self.cfg, "inverse_emission_low_info_floor", 0.02))
        floor = max(0.0, min(0.20, floor))
        observed_factor = observed.map({"no_score": floor, "unknown": floor * 0.25}).fillna(1.0)
        resolution_factor = resolution.map(
            {
                "direct_finish_signal": 1.0,
                "inferred_from_score": 0.75,
                "no_score_rule": floor,
                "ambiguous": floor * 0.20,
                "unknown": floor * 0.10,
            }
        ).fillna(max(floor * 0.25, 0.01))
        confidence_factor = confidence.map({"high": 1.0, "medium": 0.80, "low": 0.45}).fillna(0.45)
        weights = observed_factor * resolution_factor * confidence_factor
        return np.clip(weights.to_numpy(dtype=float), 0.0, 1.0)

    def _anchor_prior_matrix(self, features: pd.DataFrame, lengths: list[int] | None = None) -> np.ndarray:
        n_rows = len(features)
        out = np.ones((n_rows, self.n_states), dtype=float)
        if n_rows == 0:
            return out

        a1 = pd.to_numeric(features.get("anchor_s1"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        a2 = pd.to_numeric(features.get("anchor_s2"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        a3 = pd.to_numeric(features.get("anchor_s3"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        duration_bin = pd.to_numeric(features.get("duration_bin"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        duration_norm = duration_bin / max(1.0, float(np.max(duration_bin)))
        positions = self._row_positions(lengths or [], n_rows)

        base = np.full((n_rows, self.n_states), _EPS, dtype=float)
        if self.n_states >= 1:
            base[:, 0] = np.maximum(0.0, a1) + _EPS
            base[:, 0] *= (1.15 - 0.20 * duration_norm)
        if self.n_states >= 2:
            base[:, 1] = np.maximum(0.0, a2) + _EPS
            base[:, 1] *= (1.00 + 0.10 * duration_norm)
        if self.n_states >= 3:
            base[:, 2] = np.maximum(0.0, a3) + _EPS
            base[:, 2] *= (0.85 + 0.25 * duration_norm)
        if self.n_states > 3:
            tail = 0.35 * (np.maximum(0.0, a2) + np.maximum(0.0, a3)) / float(self.n_states - 2)
            for state_id in range(3, self.n_states):
                base[:, state_id] = np.maximum(base[:, state_id], tail + _EPS)

        base = base / np.maximum(base.sum(axis=1, keepdims=True), _EPS)
        anchor_power = float(getattr(self.cfg, "inverse_anchor_power", 2.0))
        anchor_power = max(1.0, min(4.0, anchor_power))
        base = np.power(base, anchor_power)
        base = base / np.maximum(base.sum(axis=1, keepdims=True), _EPS)

        stage = np.ones((n_rows, self.n_states), dtype=float)
        if self.n_states >= 1:
            stage[:, 0] += 1.6 * (1.0 - positions)
        if self.n_states >= 2:
            stage[:, 1] += 1.4 * (1.0 - np.clip(np.abs(positions - 0.5) * 2.0, 0.0, 1.0))
        if self.n_states >= 3:
            stage[:, 2] += 1.6 * positions
        stage = stage / np.maximum(stage.sum(axis=1, keepdims=True), _EPS)

        stage_blend = float(getattr(self.cfg, "inverse_stage_prior_blend", 0.30))
        stage_blend = max(0.0, min(0.60, stage_blend))
        stage_blend_low_info = float(getattr(self.cfg, "inverse_stage_low_info_blend", 0.75))
        stage_blend_low_info = max(stage_blend, min(0.90, stage_blend_low_info))
        resolution = (
            pd.Series(features.get("observation_resolution_type", "unknown"), index=features.index)
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )
        low_info = resolution.isin({"no_score_rule", "ambiguous", "unknown"}).to_numpy(dtype=bool)
        stage_blend_vec = np.full(n_rows, stage_blend, dtype=float)
        stage_blend_vec[low_info] = np.maximum(stage_blend_vec[low_info], stage_blend_low_info)
        out = (1.0 - stage_blend_vec.reshape(-1, 1)) * base + stage_blend_vec.reshape(-1, 1) * stage
        out = out / np.maximum(out.sum(axis=1, keepdims=True), _EPS)
        return out

    def _blend_gamma(self, gamma: np.ndarray, anchor_prior: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        eps = 1e-12
        alpha = float(np.clip(alpha, 0.0, 1.0))
        anchored = gamma * anchor_prior
        anchored /= np.maximum(anchored.sum(axis=1, keepdims=True), eps)
        blended = (1.0 - alpha) * gamma + alpha * anchored
        blended /= np.maximum(blended.sum(axis=1, keepdims=True), eps)
        return blended

    def _row_positions(self, lengths: list[int], total_rows: int) -> np.ndarray:
        if total_rows <= 0:
            return np.array([], dtype=float)
        if not lengths:
            if total_rows == 1:
                return np.array([0.0], dtype=float)
            return np.linspace(0.0, 1.0, total_rows)

        positions = np.zeros(total_rows, dtype=float)
        start = 0
        for length in lengths:
            length = int(length)
            if length <= 0:
                continue
            end = start + length
            if length == 1:
                positions[start:end] = 0.0
            else:
                positions[start:end] = np.linspace(0.0, 1.0, length)
            start = end
        return positions

    def _finish_distance(self, observed_labels: list[str], lengths: list[int]) -> np.ndarray:
        total = len(observed_labels)
        if total == 0:
            return np.array([], dtype=float)

        if not lengths:
            lengths = [total]
        labels = [str(x) for x in observed_labels]
        out = np.ones(total, dtype=float)
        start = 0
        for length in lengths:
            length = int(length)
            if length <= 0:
                continue
            end = start + length
            seq_labels = labels[start:end]
            finish_idx = [i for i, name in enumerate(seq_labels) if name in _FINISH_LIKE_OBSERVED_CLASSES]
            if finish_idx:
                denom = float(max(1, length - 1))
                for local_idx in range(length):
                    nearest = min(abs(local_idx - fidx) for fidx in finish_idx)
                    out[start + local_idx] = float(nearest / denom)
            else:
                out[start:end] = 1.0
            start = end
        return out

    def _derive_inverse_semantic_ordering(
        self,
        semantic_features: pd.DataFrame,
        decoded_states: np.ndarray,
        observed_labels: list[str],
        lengths: list[int],
    ) -> SemanticOrderingDiagnostics:
        n_states = self.n_states
        frame = semantic_features.copy().reset_index(drop=True)
        frame["_state"] = pd.Series(decoded_states).astype(int)
        frame["_position"] = self._row_positions(lengths, len(frame))
        frame["_finish_distance"] = self._finish_distance(observed_labels, lengths)
        frame["_productive_obs"] = pd.Series(observed_labels).isin(_FINISH_LIKE_OBSERVED_CLASSES).astype(float)

        transition_counts = np.zeros((n_states, n_states), dtype=float)
        start_counts = np.zeros(n_states, dtype=float)
        end_counts = np.zeros(n_states, dtype=float)
        effective_lengths = lengths or [len(decoded_states)]
        start = 0
        for length in effective_lengths:
            length = int(length)
            if length <= 0:
                continue
            end = start + length
            seq_states = np.asarray(decoded_states[start:end], dtype=int)
            if seq_states.size == 0:
                start = end
                continue
            start_counts[int(seq_states[0])] += 1.0
            end_counts[int(seq_states[-1])] += 1.0
            for t in range(len(seq_states) - 1):
                src = int(seq_states[t])
                dst = int(seq_states[t + 1])
                if 0 <= src < n_states and 0 <= dst < n_states:
                    transition_counts[src, dst] += 1.0
            start = end

        grouped = frame.groupby("_state", dropna=False)
        count_total = float(max(1, len(frame)))

        state_rows: list[dict[str, object]] = []
        for state_id in range(n_states):
            if state_id in grouped.groups:
                row_frame = grouped.get_group(state_id)
            else:
                row_frame = frame.iloc[0:0]

            if row_frame.empty:
                coverage = 0.0
                anchor_s1 = 0.0
                anchor_s2 = 0.0
                anchor_s3 = 0.0
                position_mean = 0.0
                finish_distance = 1.0
                productive_share = 0.0
                duration_mean = 0.0
                start_share = 0.0
                end_share = 0.0
                incoming_from_prev = 0.0
                outgoing_to_next = 0.0
                self_transition_share = 0.0
                bridge_role = 0.0
            else:
                coverage = float(len(row_frame) / count_total)
                anchor_s1 = float(pd.to_numeric(row_frame["anchor_s1"], errors="coerce").fillna(0.0).mean())
                anchor_s2 = float(pd.to_numeric(row_frame["anchor_s2"], errors="coerce").fillna(0.0).mean())
                anchor_s3 = float(pd.to_numeric(row_frame["anchor_s3"], errors="coerce").fillna(0.0).mean())
                position_mean = float(pd.to_numeric(row_frame["_position"], errors="coerce").fillna(0.0).mean())
                finish_distance = float(pd.to_numeric(row_frame["_finish_distance"], errors="coerce").fillna(1.0).mean())
                productive_share = float(pd.to_numeric(row_frame["_productive_obs"], errors="coerce").fillna(0.0).mean())
                duration_mean = float(pd.to_numeric(row_frame["duration_bin"], errors="coerce").fillna(0.0).mean())
                transitions_out = float(transition_counts[state_id, :].sum())
                transitions_in = float(transition_counts[:, state_id].sum())
                start_share = float(start_counts[state_id] / max(1.0, float(start_counts.sum())))
                end_share = float(end_counts[state_id] / max(1.0, float(end_counts.sum())))
                incoming_from_prev = (
                    float(transition_counts[state_id - 1, state_id] / max(1.0, transitions_in))
                    if state_id > 0
                    else 0.0
                )
                outgoing_to_next = (
                    float(transition_counts[state_id, state_id + 1] / max(1.0, transitions_out))
                    if state_id + 1 < n_states
                    else 0.0
                )
                self_transition_share = float(transition_counts[state_id, state_id] / max(1.0, transitions_out))
                bridge_role = 0.5 * incoming_from_prev + 0.5 * outgoing_to_next

            state_rows.append(
                {
                    "state_id": int(state_id),
                    "coverage_share": coverage,
                    "anchor_s1_mean": anchor_s1,
                    "anchor_s2_mean": anchor_s2,
                    "anchor_s3_mean": anchor_s3,
                    "position_mean": position_mean,
                    "finish_distance_mean": finish_distance,
                    "productive_observation_share": productive_share,
                    "duration_bin_mean": duration_mean,
                    "start_share": start_share,
                    "end_share": end_share,
                    "incoming_from_prev_share": incoming_from_prev,
                    "outgoing_to_next_share": outgoing_to_next,
                    "self_transition_share": self_transition_share,
                    "transition_bridge_role": bridge_role,
                }
            )

        anchor_s1_all = np.array([float(row["anchor_s1_mean"]) for row in state_rows], dtype=float)
        anchor_s2_all = np.array([float(row["anchor_s2_mean"]) for row in state_rows], dtype=float)
        anchor_s3_all = np.array([float(row["anchor_s3_mean"]) for row in state_rows], dtype=float)
        coverage_all = np.array([float(row["coverage_share"]) for row in state_rows], dtype=float)
        position_all = np.array([float(row["position_mean"]) for row in state_rows], dtype=float)
        finish_dist_all = np.array([float(row["finish_distance_mean"]) for row in state_rows], dtype=float)
        finish_close_all = 1.0 - np.clip(finish_dist_all, 0.0, 1.0)
        productive_all = np.array([float(row["productive_observation_share"]) for row in state_rows], dtype=float)
        start_share_all = np.array([float(row["start_share"]) for row in state_rows], dtype=float)
        end_share_all = np.array([float(row["end_share"]) for row in state_rows], dtype=float)
        bridge_role_all = np.array([float(row["transition_bridge_role"]) for row in state_rows], dtype=float)

        anchor_s1_norm = self._minmax(anchor_s1_all)
        anchor_s2_norm = self._minmax(anchor_s2_all)
        anchor_s3_norm = self._minmax(anchor_s3_all)
        coverage_norm = self._minmax(coverage_all)
        position_norm = self._minmax(position_all)
        finish_close_norm = self._minmax(finish_close_all)
        productive_norm = self._minmax(productive_all)
        start_norm = self._minmax(start_share_all)
        end_norm = self._minmax(end_share_all)
        bridge_norm = self._minmax(bridge_role_all)
        middle_position = 1.0 - np.clip(np.abs(position_norm - 0.5) * 2.0, 0.0, 1.0)

        score_by_semantic: dict[str, np.ndarray] = {
            "S1": (
                0.42 * anchor_s1_norm
                + 0.20 * (1.0 - position_norm)
                + 0.20 * start_norm
                + 0.10 * coverage_norm
                + 0.08 * (1.0 - np.clip(finish_dist_all, 0.0, 1.0))
            ),
            "S2": (
                0.38 * anchor_s2_norm
                + 0.22 * middle_position
                + 0.20 * bridge_norm
                + 0.12 * coverage_norm
                + 0.08 * (1.0 - np.clip(finish_dist_all, 0.0, 1.0))
            ),
            "S3": (
                0.35 * anchor_s3_norm
                + 0.20 * position_norm
                + 0.20 * finish_close_norm
                + 0.17 * end_norm
                + 0.08 * productive_norm
            ),
        }
        anchor_order_by_state: dict[int, list[tuple[str, float]]] = {}
        anchor_margin_by_state: dict[int, float] = {}
        for state_id in range(n_states):
            ranking = sorted(
                [
                    ("S1", float(state_rows[state_id]["anchor_s1_mean"])),
                    ("S2", float(state_rows[state_id]["anchor_s2_mean"])),
                    ("S3", float(state_rows[state_id]["anchor_s3_mean"])),
                ],
                key=lambda it: float(it[1]),
                reverse=True,
            )
            anchor_order_by_state[state_id] = ranking
            anchor_margin_by_state[state_id] = (
                float(ranking[0][1] - ranking[1][1]) if len(ranking) > 1 else float(ranking[0][1])
            )

        # Weak supervision guardrail:
        # keep semantic assignment close to anchor dominance when the anchor signal is clear.
        for semantic_name in _SEMANTIC_NAMES:
            for state_id in range(n_states):
                ranking = anchor_order_by_state[state_id]
                margin = float(anchor_margin_by_state[state_id])
                rank_labels = [item[0] for item in ranking]
                try:
                    rank = rank_labels.index(semantic_name)
                except ValueError:
                    rank = 2
                if rank == 0:
                    bonus = 0.10 + 0.25 * np.clip(margin, 0.0, 0.35)
                elif rank == 1:
                    bonus = 0.02
                else:
                    bonus = -0.12 - 0.20 * np.clip(margin, 0.0, 0.35)
                score_by_semantic[semantic_name][state_id] = float(
                    score_by_semantic[semantic_name][state_id] + bonus
                )

        semantic_to_state: dict[str, int] = {}
        semantic_confidence: dict[str, float] = {name: 0.0 for name in _SEMANTIC_NAMES}
        warnings: list[str] = []
        observed_norm = pd.Series(observed_labels).fillna("unknown").astype(str).str.lower()
        informative_share = float((~observed_norm.isin({"no_score", "unknown"})).mean()) if len(observed_norm) else 0.0
        informative_scale = float(np.clip((informative_share + 0.05) / 0.30, 0.25, 1.0))

        state_ids = list(range(n_states))
        if n_states >= 3:
            best_perm: tuple[int, int, int] | None = None
            best_score = -1e18
            for candidate in permutations(state_ids, 3):
                score = (
                    float(score_by_semantic["S1"][candidate[0]])
                    + float(score_by_semantic["S2"][candidate[1]])
                    + float(score_by_semantic["S3"][candidate[2]])
                )
                if score > best_score:
                    best_score = score
                    best_perm = candidate
            if best_perm is not None:
                semantic_to_state = {"S1": int(best_perm[0]), "S2": int(best_perm[1]), "S3": int(best_perm[2])}
        else:
            remaining = set(state_ids)
            for semantic_name in _SEMANTIC_NAMES[:n_states]:
                ranked = sorted(
                    list(remaining),
                    key=lambda sid: float(score_by_semantic[semantic_name][sid]),
                    reverse=True,
                )
                if not ranked:
                    continue
                sid = int(ranked[0])
                semantic_to_state[semantic_name] = sid
                remaining.discard(sid)

        for semantic_name, state_id in semantic_to_state.items():
            scores = score_by_semantic[semantic_name]
            best = float(scores[state_id])
            alternatives = [float(scores[idx]) for idx in state_ids if idx != state_id]
            second = max(alternatives) if alternatives else 0.0
            margin = max(0.0, best - second)
            coverage = float(state_rows[state_id]["coverage_share"])
            coverage_term = float(np.clip(coverage / 0.12, 0.0, 1.0))
            if semantic_name == "S3":
                coverage_term = max(coverage_term, float(finish_close_norm[state_id]))
            ranking = anchor_order_by_state[state_id]
            dominant_anchor = str(ranking[0][0]) if ranking else ""
            anchor_margin = float(anchor_margin_by_state.get(state_id, 0.0))
            anchor_match = bool(dominant_anchor == semantic_name)
            anchor_term = 1.0 if anchor_match else float(np.clip(1.0 - 3.0 * anchor_margin, 0.0, 1.0))
            conf = float(np.clip(0.45 * best + 0.25 * margin + 0.15 * coverage_term + 0.15 * anchor_term, 0.0, 1.0))
            if not anchor_match and anchor_margin >= 0.03:
                conf *= 0.72
                warnings.append(
                    f"{semantic_name} assignment conflicts with dominant anchor "
                    f"({dominant_anchor}, margin={anchor_margin:.3f}); confidence penalized."
                )
            if semantic_name in {"S1", "S2"} and coverage < 0.05:
                conf *= 0.70
                warnings.append(
                    f"{semantic_name} assigned state has low coverage ({coverage:.3f}); interpretation may be unstable."
                )
            if semantic_name == "S3" and float(finish_close_norm[state_id]) < 0.40:
                conf *= 0.75
                warnings.append(
                    "S3 assigned state is not close enough to finish-like observations; confidence penalized."
                )
            conf *= informative_scale
            conf = float(np.clip(conf, 0.0, 1.0))
            semantic_confidence[semantic_name] = conf
            if conf < 0.35:
                warnings.append(
                    f"{semantic_name} assignment confidence is low ({conf:.3f}); semantic interpretation may be unstable."
                )

        if set(semantic_to_state.keys()) != set(_SEMANTIC_NAMES):
            warnings.append("Not all semantic states (S1/S2/S3) were assigned.")

        name_mapping: dict[int, str] = {}
        desc_mapping: dict[int, str] = {}
        reverse_semantic = {int(v): str(k) for k, v in semantic_to_state.items()}
        for state_id in range(n_states):
            semantic_name = reverse_semantic.get(state_id, "")
            if semantic_name:
                name_mapping[state_id] = semantic_name
                if semantic_name == "S1":
                    desc_mapping[state_id] = "Маневрирование/стойки: входной сегмент диагностической цепочки."
                elif semantic_name == "S2":
                    desc_mapping[state_id] = "КФВ: первичная причинная зона в S1→S2→S3."
                else:
                    desc_mapping[state_id] = "ВУП: вторичная причинная зона, ближайшая к результативным наблюдениям."
            else:
                name_mapping[state_id] = f"state_{state_id}"
                desc_mapping[state_id] = "Нейтральное латентное состояние без устойчивой семантической привязки."

        semantic_prefix = [semantic_to_state[name] for name in _SEMANTIC_NAMES if name in semantic_to_state]
        trailing = [sid for sid in range(n_states) if sid not in semantic_prefix]
        canonical_order = semantic_prefix + trailing

        for row in state_rows:
            sid = int(row["state_id"])
            row["assigned_name"] = name_mapping.get(sid, f"state_{sid}")
            row["semantic_score_s1"] = float(score_by_semantic["S1"][sid])
            row["semantic_score_s2"] = float(score_by_semantic["S2"][sid])
            row["semantic_score_s3"] = float(score_by_semantic["S3"][sid])
            dominant = max(
                [("S1", row["anchor_s1_mean"]), ("S2", row["anchor_s2_mean"]), ("S3", row["anchor_s3_mean"])],
                key=lambda x: float(x[1]),
            )
            row["dominant_anchor"] = dominant[0]
            row["dominant_anchor_value"] = float(dominant[1])

        semantic_order_matches_topology = canonical_order == list(range(n_states))
        if not semantic_order_matches_topology:
            warnings.append("Semantic order differs from internal topology order before canonical reordering.")

        if {"S1", "S2", "S3"}.issubset(set(semantic_to_state)):
            if not (
                semantic_to_state["S1"] <= semantic_to_state["S2"] <= semantic_to_state["S3"]
            ):
                warnings.append("Assigned S1/S2/S3 order conflicts with left-to-right progression.")
        if informative_share < 0.15:
            warnings.append(
                "Observed sequence has low informative share (few non-no_score/unknown observations); "
                "semantic assignment confidence is intentionally downscaled."
            )

        return SemanticOrderingDiagnostics(
            original_name_mapping=name_mapping,
            original_description_mapping=desc_mapping,
            canonical_order=[int(x) for x in canonical_order],
            semantic_to_original_state={k: int(v) for k, v in semantic_to_state.items()},
            semantic_confidence={k: float(v) for k, v in semantic_confidence.items()},
            state_profiles=state_rows,
            semantic_order_matches_topology=semantic_order_matches_topology,
            warnings=warnings,
        )

    def fit(
        self,
        observed_sequence: pd.Series | list[str],
        sequence_ids: pd.Series | None = None,
        hidden_state_features: pd.DataFrame | None = None,
    ) -> float:
        observed_labels = (
            observed_sequence.fillna("unknown").astype(str).tolist()
            if isinstance(observed_sequence, pd.Series)
            else ["unknown" if v is None else str(v) for v in observed_sequence]
        )
        obs_idx = self._encode_observations(observed_labels)
        lengths = self._sequence_lengths(sequence_ids)
        sequences = self._split_by_lengths(obs_idx, lengths)

        if hidden_state_features is not None and len(hidden_state_features) == len(obs_idx):
            semantic_features = self._semantic_feature_view(hidden_state_features)
        else:
            semantic_features = self._semantic_feature_view(pd.DataFrame(index=range(len(obs_idx))))

        row_weights = self._row_train_weights(semantic_features, observed_labels=observed_labels)
        emission_row_weights = self._row_emission_weights(semantic_features, observed_labels=observed_labels)
        anchor_prior = self._anchor_prior_matrix(semantic_features, lengths=lengths)
        weight_sequences = self._split_by_lengths(row_weights, lengths)
        emission_weight_sequences = self._split_by_lengths(emission_row_weights, lengths)
        anchor_sequences = self._split_by_lengths(anchor_prior, lengths)

        self._initialize_parameters()
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.emissionprob_ is not None

        smoothing = 1e-3
        tol = 1e-4
        max_iter = max(1, int(getattr(self.cfg, "n_iter", 100)))
        anchor_alpha = float(getattr(self.cfg, "inverse_anchor_blend", 0.35))
        anchor_obs_strength = float(getattr(self.cfg, "inverse_anchor_observation_strength", 0.55))
        anchor_obs_strength = max(0.0, min(1.5, anchor_obs_strength))
        state_balance_prior_strength = float(getattr(self.cfg, "inverse_state_balance_prior_strength", 0.15))
        state_balance_prior_strength = max(0.0, min(1.0, state_balance_prior_strength))
        emission_prior_blend = float(getattr(self.cfg, "inverse_emission_prior_blend", 0.08))

        prev_ll: float | None = None
        converged = False
        last_delta: float | None = None
        ll = float("-inf")

        for it in range(1, max_iter + 1):
            start_counts = np.zeros(self.n_states, dtype=float)
            trans_counts = np.zeros((self.n_states, self.n_states), dtype=float)
            emission_counts = np.zeros((self.n_states, self.n_observations), dtype=float)
            anchor_emission_prior = np.zeros((self.n_states, self.n_observations), dtype=float)

            ll = 0.0
            for seq, seq_weights, seq_emission_weights, seq_anchor in zip(
                sequences,
                weight_sequences,
                emission_weight_sequences,
                anchor_sequences,
            ):
                seq_ll, gamma, _, xi_steps = self._forward_backward(
                    seq,
                    state_prior=seq_anchor,
                    state_prior_strength=anchor_obs_strength,
                )
                ll += seq_ll
                gamma_adj = self._blend_gamma(gamma, seq_anchor, alpha=anchor_alpha)

                if len(seq_weights):
                    start_counts += gamma_adj[0] * max(float(seq_weights[0]), _EPS)
                else:
                    start_counts += gamma_adj[0]

                if len(seq_weights) > 1 and len(xi_steps):
                    left = np.maximum(seq_weights[:-1], 0.0)
                    right = np.maximum(seq_weights[1:], 0.0)
                    pair_weights = np.sqrt(left * right)
                    if float(pair_weights.sum()) > 0.0:
                        trans_counts += np.tensordot(pair_weights, xi_steps, axes=(0, 0))
                    else:
                        trans_counts += np.sum(xi_steps, axis=0) * _EPS
                elif len(xi_steps):
                    trans_counts += np.sum(xi_steps, axis=0) * _EPS

                for t, obs in enumerate(seq):
                    obs_weight = max(float(seq_weights[t]), 0.0)
                    emission_weight = max(float(seq_emission_weights[t]), 0.0)
                    emission_counts[:, obs] += gamma_adj[t] * obs_weight * emission_weight
                    anchor_emission_prior[:, obs] += seq_anchor[t] * obs_weight

            if state_balance_prior_strength > 0.0:
                emission_counts += state_balance_prior_strength * anchor_emission_prior

            start_new = self._normalize(start_counts + smoothing)
            if self.cfg.topology_mode == "left_to_right":
                start_new = np.zeros(self.n_states, dtype=float)
                start_new[0] = 1.0

            trans_new = self._enforce_topology(trans_counts + smoothing)
            emission_new = self._normalize_rows(emission_counts + smoothing)
            if emission_prior_blend > 0.0:
                prior = self._semantic_emission_prior()
                blend = float(np.clip(emission_prior_blend, 0.0, 0.5))
                emission_new = self._normalize_rows((1.0 - blend) * emission_new + blend * prior)

            self.startprob_ = start_new
            self.transmat_ = trans_new
            self.emissionprob_ = emission_new

            if prev_ll is not None:
                delta = float(ll - prev_ll)
                last_delta = delta
                if abs(delta) < tol:
                    converged = True
                    break
            prev_ll = ll

        self.last_training_result = InverseTrainingResult(
            log_likelihood=float(ll),
            converged=converged,
            n_iterations=it,
            last_delta=last_delta,
        )

        if hidden_state_features is not None and len(hidden_state_features) == len(obs_idx):
            infer_anchor_strength = float(getattr(self.cfg, "inverse_inference_anchor_strength", 0.85))
            infer_anchor_strength = max(0.0, min(1.5, infer_anchor_strength))
            states, _ = self._decode_states(
                obs_idx,
                lengths=lengths,
                state_prior=anchor_prior,
                state_prior_strength=infer_anchor_strength,
            )
            semantic_diag = self._derive_inverse_semantic_ordering(
                semantic_features=semantic_features,
                decoded_states=states,
                observed_labels=observed_labels,
                lengths=lengths,
            )

            canonical_order = [int(x) for x in semantic_diag.canonical_order]
            reordered = False
            if canonical_order != list(range(self.n_states)) and getattr(self.cfg, "canonical_reorder_enabled", True):
                self._reorder_states(canonical_order)
                reordered = True

            if reordered:
                # Recompute semantic diagnostics on the reordered model so coverage,
                # role metrics and confidence are aligned with final inference states.
                states_post, _ = self._decode_states(
                    obs_idx,
                    lengths=lengths,
                    state_prior=anchor_prior,
                    state_prior_strength=infer_anchor_strength,
                )
                semantic_post = self._derive_inverse_semantic_ordering(
                    semantic_features=semantic_features,
                    decoded_states=states_post,
                    observed_labels=observed_labels,
                    lengths=lengths,
                )
                warnings = list(semantic_post.warnings)
                if semantic_post.canonical_order != list(range(self.n_states)):
                    warnings.append(
                        "Post-reorder semantic ordering still differs from topology; "
                        "keeping current state order and exposing low-confidence semantics."
                    )
                semantic_diag = SemanticOrderingDiagnostics(
                    original_name_mapping=semantic_post.original_name_mapping,
                    original_description_mapping=semantic_post.original_description_mapping,
                    canonical_order=list(range(self.n_states)),
                    semantic_to_original_state=semantic_post.semantic_to_original_state,
                    semantic_confidence=semantic_post.semantic_confidence,
                    state_profiles=semantic_post.state_profiles,
                    semantic_order_matches_topology=True,
                    warnings=warnings,
                )

            self.state_definition = semantic_diag.canonical_state_definition(self.n_states)
            self.last_semantic_diagnostics = self._canonical_semantic_payload(semantic_diag)
            self.last_canonical_state_mapping = self._build_canonical_state_mapping()
        else:
            self.last_semantic_diagnostics = {
                "canonical_order_from_original": list(range(self.n_states)),
                "semantic_to_state": {},
                "semantic_confidence": {},
                "state_profiles": [],
                "warnings": ["semantic_diagnostics_skipped_due_to_missing_hidden_features"],
                "semantic_order_matches_topology_before_reorder": True,
            }
            self.last_canonical_state_mapping = self._build_canonical_state_mapping()

        logger.info(
            "Inverse HMM fitted. log_likelihood=%.4f, converged=%s, n_iter=%d, last_delta=%s",
            ll,
            converged,
            it,
            None if last_delta is None else round(float(last_delta), 6),
        )
        return float(ll)

    def _decode_states(
        self,
        obs_idx: np.ndarray,
        lengths: list[int] | None = None,
        *,
        state_prior: np.ndarray | None = None,
        state_prior_strength: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        sequences = self._split_by_lengths(obs_idx, lengths or [])
        prior_sequences = self._split_by_lengths(state_prior, lengths or []) if state_prior is not None else [None] * len(sequences)
        states_all: list[np.ndarray] = []
        ll = 0.0
        for seq, seq_prior in zip(sequences, prior_sequences):
            states, seq_ll = self._viterbi(
                seq,
                state_prior=seq_prior,
                state_prior_strength=state_prior_strength,
            )
            states_all.append(states)
            ll += seq_ll
        return np.concatenate(states_all, axis=0), float(ll)

    def _posterior_probabilities(
        self,
        obs_idx: np.ndarray,
        lengths: list[int] | None = None,
        *,
        state_prior: np.ndarray | None = None,
        state_prior_strength: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        sequences = self._split_by_lengths(obs_idx, lengths or [])
        prior_sequences = self._split_by_lengths(state_prior, lengths or []) if state_prior is not None else [None] * len(sequences)
        posteriors: list[np.ndarray] = []
        ll = 0.0
        for seq, seq_prior in zip(sequences, prior_sequences):
            seq_ll, gamma, _, _ = self._forward_backward(
                seq,
                state_prior=seq_prior,
                state_prior_strength=state_prior_strength,
            )
            posteriors.append(gamma)
            ll += seq_ll
        return np.concatenate(posteriors, axis=0), float(ll)

    def predict(
        self,
        observed_sequence: pd.Series | list[str],
        sequence_ids: pd.Series | None = None,
        hidden_state_features: pd.DataFrame | None = None,
    ) -> InverseHMMPrediction:
        if self.startprob_ is None or self.transmat_ is None or self.emissionprob_ is None:
            raise ValueError("Model is not fitted yet.")

        obs_idx = self._encode_observations(observed_sequence)
        lengths = self._sequence_lengths(sequence_ids)
        state_prior = None
        state_prior_strength = 0.0
        if hidden_state_features is not None and len(hidden_state_features) == len(obs_idx):
            semantic_features = self._semantic_feature_view(hidden_state_features)
            state_prior = self._anchor_prior_matrix(semantic_features, lengths=lengths)
            state_prior_strength = float(getattr(self.cfg, "inverse_inference_anchor_strength", 0.85))
            state_prior_strength = max(0.0, min(1.5, state_prior_strength))
        states, _ = self._decode_states(
            obs_idx,
            lengths=lengths,
            state_prior=state_prior,
            state_prior_strength=state_prior_strength,
        )
        posterior, log_likelihood = self._posterior_probabilities(
            obs_idx,
            lengths=lengths,
            state_prior=state_prior,
            state_prior_strength=state_prior_strength,
        )

        return InverseHMMPrediction(
            states=states,
            state_probabilities=posterior,
            log_likelihood=float(log_likelihood),
            state_names=[self.state_definition.state_name(int(s)) for s in states],
        )

    def canonical_state_mapping(self) -> dict[str, object]:
        if self.last_canonical_state_mapping is not None:
            return self.last_canonical_state_mapping
        self.last_canonical_state_mapping = self._build_canonical_state_mapping()
        return self.last_canonical_state_mapping

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "observation_classes": self.observation_classes,
            "state_definition": self.state_definition.to_dict(),
            "startprob": self.startprob_,
            "transmat": self.transmat_,
            "emissionprob": self.emissionprob_,
            "semantic_diagnostics": self.last_semantic_diagnostics,
            "canonical_state_mapping": self.last_canonical_state_mapping,
            "training_result": self.last_training_result,
        }
        with file_path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "InverseDiagnosticHMM":
        file_path = Path(path)
        with file_path.open("rb") as f:
            payload = pickle.load(f)

        state_definition = StateDefinition.from_dict(payload.get("state_definition", {}))
        if not state_definition.states:
            state_definition = StateDefinition.research_default(payload["cfg"].n_hidden_states)

        obj = cls(
            cfg=payload["cfg"],
            observation_classes=[str(x) for x in payload.get("observation_classes", [])],
            state_definition=state_definition,
        )
        obj.startprob_ = payload.get("startprob")
        obj.transmat_ = payload.get("transmat")
        obj.emissionprob_ = payload.get("emissionprob")
        obj.last_semantic_diagnostics = payload.get("semantic_diagnostics")
        obj.last_canonical_state_mapping = payload.get("canonical_state_mapping")
        obj.last_training_result = payload.get("training_result")

        if obj.last_canonical_state_mapping is None:
            obj.last_canonical_state_mapping = obj._build_canonical_state_mapping()
        return obj
