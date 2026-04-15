from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from hidden_patterns_combat.config import ModelConfig
from hidden_patterns_combat.modeling.observation_encoding import build_lengths
from hidden_patterns_combat.modeling.state_definition import (
    SemanticOrderingDiagnostics,
    StateDefinition,
    derive_semantic_ordering,
)

logger = logging.getLogger(__name__)


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

    Training uses observed ZAP classes. Hidden-state features are optional and are used
    only for semantic initialization/relabeling diagnostics.
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
            return np.ones_like(v) / len(v)
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

    def _init_startprob_transmat(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_states
        if self.cfg.topology_mode == "left_to_right":
            start = np.zeros(n, dtype=float)
            start[0] = 1.0

            trans = np.zeros((n, n), dtype=float)
            for i in range(n):
                allowed = list(range(i, n))
                for j in allowed:
                    trans[i, j] = 1.0
            trans = self._normalize_rows(trans)
            return start, trans

        start = np.ones(n, dtype=float) / float(n)
        trans = np.ones((n, n), dtype=float)
        trans = self._normalize_rows(trans)
        return start, trans

    def _semantic_emission_prior(self) -> np.ndarray:
        n = self.n_states
        m = self.n_observations
        emissions = np.ones((n, m), dtype=float)

        class_idx = self.obs_to_idx

        def bump(state: int, class_name: str, weight: float) -> None:
            idx = class_idx.get(class_name)
            if idx is None:
                return
            emissions[state, idx] += weight

        if n >= 1:
            bump(0, "no_score", 6.0)
            bump(0, "zap_r", 3.0)
            bump(0, "unknown", 2.0)

        if n >= 2:
            bump(1, "hold", 4.0)
            bump(1, "zap_n", 3.0)
            bump(1, "zap_r", 2.0)

        if n >= 3:
            bump(2, "zap_t", 4.0)
            bump(2, "arm_submission", 4.0)
            bump(2, "leg_submission", 4.0)
            bump(2, "zap_n", 2.0)

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

        if "unknown" in self.obs_to_idx:
            unknown_idx = self.obs_to_idx["unknown"]
        else:
            unknown_idx = 0

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

    def _forward_backward(self, obs: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.emissionprob_ is not None

        n = self.n_states
        t_max = len(obs)
        eps = 1e-12

        alpha = np.zeros((t_max, n), dtype=float)
        beta = np.zeros((t_max, n), dtype=float)
        scales = np.zeros(t_max, dtype=float)

        alpha[0] = self.startprob_ * self.emissionprob_[:, obs[0]]
        scales[0] = max(float(alpha[0].sum()), eps)
        alpha[0] /= scales[0]

        for t in range(1, t_max):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * self.emissionprob_[:, obs[t]]
            scales[t] = max(float(alpha[t].sum()), eps)
            alpha[t] /= scales[t]

        beta[-1] = np.ones(n, dtype=float)
        for t in range(t_max - 2, -1, -1):
            beta[t] = (self.transmat_ * self.emissionprob_[:, obs[t + 1]].reshape(1, -1)) @ beta[t + 1]
            beta[t] /= max(scales[t + 1], eps)

        gamma = alpha * beta
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), eps)

        xi_sum = np.zeros((n, n), dtype=float)
        for t in range(t_max - 1):
            numer = (
                alpha[t].reshape(-1, 1)
                * self.transmat_
                * self.emissionprob_[:, obs[t + 1]].reshape(1, -1)
                * beta[t + 1].reshape(1, -1)
            )
            denom = max(float(numer.sum()), eps)
            xi_sum += numer / denom

        log_likelihood = float(np.sum(np.log(np.maximum(scales, eps))))
        return log_likelihood, gamma, xi_sum

    def _viterbi(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
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

        delta[0] = log_start + log_emit[:, obs[0]]

        for t in range(1, t_max):
            scores = delta[t - 1].reshape(-1, 1) + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emit[:, obs[t]]

        states = np.zeros(t_max, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(t_max - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])

        return states, float(np.max(delta[-1]))

    def _enforce_topology(self, trans: np.ndarray) -> np.ndarray:
        out = trans.copy()
        if self.cfg.topology_mode == "left_to_right":
            for i in range(out.shape[0]):
                out[i, :i] = 0.0

            min_forward = float(getattr(self.cfg, "min_forward_transition", 0.0))
            if min_forward > 0:
                for i in range(out.shape[0] - 1):
                    out[i, i + 1] = max(out[i, i + 1], min_forward)
        out = self._normalize_rows(out)
        return out

    def _canonical_semantic_payload(
        self,
        semantic_diag: SemanticOrderingDiagnostics,
    ) -> dict[str, object]:
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
            "semantic_confidence": {
                k: float(v) for k, v in semantic_diag.semantic_confidence.items()
            },
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

    @staticmethod
    def _semantic_feature_view(hidden_features: pd.DataFrame) -> pd.DataFrame:
        frame = hidden_features.copy().reset_index(drop=True)
        out = pd.DataFrame(index=frame.index)

        def _num_col(name: str) -> pd.Series:
            if name not in frame.columns:
                return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
            return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)

        out["maneuver_right_code"] = _num_col("maneuver_right_code")
        out["maneuver_left_code"] = _num_col("maneuver_left_code")
        out["grips_code"] = _num_col("kfv_capture_code")
        out["holds_code"] = _num_col("kfv_grip_code")
        out["bodylocks_code"] = _num_col("kfv_wrap_code")
        out["underhooks_code"] = _num_col("kfv_hook_code")
        out["posts_code"] = _num_col("kfv_post_code")
        out["kfv_code"] = (
            out["grips_code"]
            + out["holds_code"]
            + out["bodylocks_code"]
            + out["underhooks_code"]
            + out["posts_code"]
        )
        out["vup_code"] = _num_col("vup_code")
        out["duration"] = _num_col("episode_time_sec")
        out["pause"] = _num_col("pause_time_sec")
        return out

    def fit(
        self,
        observed_sequence: pd.Series | list[str],
        sequence_ids: pd.Series | None = None,
        hidden_state_features: pd.DataFrame | None = None,
    ) -> float:
        obs_idx = self._encode_observations(observed_sequence)
        lengths = self._sequence_lengths(sequence_ids)
        sequences = self._split_by_lengths(obs_idx, lengths)

        self._initialize_parameters()
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        assert self.emissionprob_ is not None

        smoothing = 1e-3
        tol = 1e-4
        max_iter = max(1, int(getattr(self.cfg, "n_iter", 100)))

        prev_ll: float | None = None
        converged = False
        last_delta: float | None = None
        ll = float("-inf")

        for it in range(1, max_iter + 1):
            start_counts = np.zeros(self.n_states, dtype=float)
            trans_counts = np.zeros((self.n_states, self.n_states), dtype=float)
            emission_counts = np.zeros((self.n_states, self.n_observations), dtype=float)

            ll = 0.0
            for seq in sequences:
                seq_ll, gamma, xi_sum = self._forward_backward(seq)
                ll += seq_ll
                start_counts += gamma[0]
                trans_counts += xi_sum

                for t, obs in enumerate(seq):
                    emission_counts[:, obs] += gamma[t]

            start_new = self._normalize(start_counts + smoothing)
            trans_new = self._enforce_topology(trans_counts + smoothing)
            emission_new = self._normalize_rows(emission_counts + smoothing)

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
            states, _ = self._decode_states(obs_idx, lengths=lengths)
            semantic_features = self._semantic_feature_view(hidden_state_features)
            semantic_diag = derive_semantic_ordering(
                features=semantic_features,
                decoded_states=states,
                n_states=self.n_states,
            )

            canonical_order = [int(x) for x in semantic_diag.canonical_order]
            if canonical_order != list(range(self.n_states)) and getattr(self.cfg, "canonical_reorder_enabled", True):
                self._reorder_states(canonical_order)

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

    def _decode_states(self, obs_idx: np.ndarray, lengths: list[int] | None = None) -> tuple[np.ndarray, float]:
        sequences = self._split_by_lengths(obs_idx, lengths or [])
        states_all: list[np.ndarray] = []
        ll = 0.0
        for seq in sequences:
            states, seq_ll = self._viterbi(seq)
            states_all.append(states)
            ll += seq_ll
        return np.concatenate(states_all, axis=0), float(ll)

    def _posterior_probabilities(self, obs_idx: np.ndarray, lengths: list[int] | None = None) -> tuple[np.ndarray, float]:
        sequences = self._split_by_lengths(obs_idx, lengths or [])
        posteriors: list[np.ndarray] = []
        ll = 0.0
        for seq in sequences:
            seq_ll, gamma, _ = self._forward_backward(seq)
            posteriors.append(gamma)
            ll += seq_ll
        return np.concatenate(posteriors, axis=0), float(ll)

    def predict(
        self,
        observed_sequence: pd.Series | list[str],
        sequence_ids: pd.Series | None = None,
    ) -> InverseHMMPrediction:
        if self.startprob_ is None or self.transmat_ is None or self.emissionprob_ is None:
            raise ValueError("Model is not fitted yet.")

        obs_idx = self._encode_observations(observed_sequence)
        lengths = self._sequence_lengths(sequence_ids)

        states, _ = self._decode_states(obs_idx, lengths=lengths)
        posterior, log_likelihood = self._posterior_probabilities(obs_idx, lengths=lengths)

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
