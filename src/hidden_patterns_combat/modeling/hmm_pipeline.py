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
from hidden_patterns_combat.modeling.state_definition import (
    BLOCK_COLUMNS,
    SEMANTIC_TARGETS,
    SemanticOrderingDiagnostics,
    StateDefinition,
    derive_semantic_ordering,
)
from hidden_patterns_combat.modeling.training import HMMTrainer, TrainingResult

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
        self.post_scale_weights: np.ndarray | None = None
        self.last_training_result: TrainingResult | None = None
        self.last_semantic_diagnostics: dict[str, object] | None = None
        self.last_canonical_state_mapping: dict[str, object] | None = None

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            return features
        missing = [c for c in self.feature_columns if c not in features.columns]
        if missing:
            raise ValueError(f"Missing required HMM feature columns: {missing}")
        return features[self.feature_columns]

    @staticmethod
    def _feature_block_name(column: str) -> str:
        if column in BLOCK_COLUMNS["maneuvering"]:
            return "maneuvering"
        if column in BLOCK_COLUMNS["kfv"]:
            return "kfv"
        if column in BLOCK_COLUMNS["vup"]:
            return "vup"
        if column in {"duration", "pause"}:
            return "temporal"
        return "other"

    def _build_post_scale_weights(self, columns: list[str]) -> np.ndarray:
        block_weights_cfg = getattr(self.cfg, "block_weights", {}) or {}
        block_weights = {
            "maneuvering": float(block_weights_cfg.get("maneuvering", 1.0)),
            "kfv": float(block_weights_cfg.get("kfv", 1.0)),
            "vup": float(block_weights_cfg.get("vup", 1.0)),
            "temporal": float(block_weights_cfg.get("temporal", 1.0)),
            "other": float(block_weights_cfg.get("other", 1.0)),
        }

        counts = {key: 0 for key in block_weights}
        blocks = [self._feature_block_name(col) for col in columns]
        for block in blocks:
            counts[block] = counts.get(block, 0) + 1

        weights = np.ones(len(columns), dtype=float)
        for idx, block in enumerate(blocks):
            block_size = max(1, counts.get(block, 1))
            weights[idx] = block_weights.get(block, 1.0) / float(np.sqrt(block_size))

        return weights

    @staticmethod
    def _safe_row_signal(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
        present = [c for c in columns if c in frame.columns]
        if not present:
            return pd.Series(np.zeros(len(frame), dtype=float), index=frame.index)
        return frame[present].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs().mean(axis=1)

    def _semantic_seed_means(self, x: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        n_states = self.cfg.n_hidden_states
        n_rows = x.shape[0]

        signals = {
            "maneuvering": self._safe_row_signal(features, BLOCK_COLUMNS["maneuvering"]),
            "kfv": self._safe_row_signal(features, BLOCK_COLUMNS["kfv"]),
            "vup": self._safe_row_signal(features, BLOCK_COLUMNS["vup"]),
        }

        means = np.zeros((n_states, x.shape[1]), dtype=float)
        row_order = np.arange(n_rows)
        chunks = np.array_split(row_order, n_states) if n_states > 0 else []
        top_n = max(3, int(round(n_rows * 0.12)))

        for state_idx in range(n_states):
            target_block = SEMANTIC_TARGETS[min(state_idx, len(SEMANTIC_TARGETS) - 1)][0]
            target_signal = signals[target_block]
            other_blocks = [b for b in signals if b != target_block]
            if other_blocks:
                other_signal = pd.concat([signals[b] for b in other_blocks], axis=1).max(axis=1)
            else:
                other_signal = pd.Series(np.zeros(n_rows, dtype=float), index=features.index)

            dominance = (target_signal - other_signal).sort_values(ascending=False)
            selected_idx = dominance[dominance > 0].head(top_n).index.to_numpy(dtype=int)

            if len(selected_idx) == 0:
                fallback = chunks[state_idx] if state_idx < len(chunks) else np.array([], dtype=int)
                if len(fallback) == 0:
                    fallback = np.arange(min(top_n, n_rows), dtype=int)
                selected_idx = fallback

            means[state_idx] = x[selected_idx].mean(axis=0)

        return means

    def _initialize_semantic_priors(self, x: np.ndarray, features: pd.DataFrame) -> None:
        if not getattr(self.cfg, "semantic_init_enabled", True):
            return

        n_states = self.cfg.n_hidden_states
        if n_states <= 0:
            return

        self.model.init_params = ""

        if self.cfg.topology_mode == "left_to_right":
            startprob, transmat = HMMTrainer.left_to_right_prior(n_states)
        else:
            startprob = np.ones(n_states, dtype=float) / n_states
            transmat = np.ones((n_states, n_states), dtype=float) / n_states
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat

        means = self._semantic_seed_means(x=x, features=features)
        self.model.means_ = means

        variances = np.var(x, axis=0) + 1e-2
        cov_type = self.cfg.covariance_type
        if cov_type == "diag":
            self.model.covars_ = np.tile(variances.reshape(1, -1), (n_states, 1))
        elif cov_type == "spherical":
            self.model.covars_ = np.ones(n_states, dtype=float) * float(np.mean(variances))
        elif cov_type == "tied":
            self.model.covars_ = np.diag(variances)
        else:  # full
            self.model.covars_ = np.repeat(np.diag(variances)[None, :, :], n_states, axis=0)

    @staticmethod
    def _inverse_permutation(order: list[int]) -> np.ndarray:
        inv = np.zeros(len(order), dtype=int)
        for new_idx, old_idx in enumerate(order):
            inv[int(old_idx)] = int(new_idx)
        return inv

    def _reorder_model_states(self, order: list[int]) -> None:
        n = self.cfg.n_hidden_states
        if sorted(order) != list(range(n)):
            raise ValueError(f"Invalid state permutation for reordering: {order}")

        perm = np.asarray(order, dtype=int)

        self.model.startprob_ = self.model.startprob_[perm]
        self.model.transmat_ = self.model.transmat_[perm][:, perm]

        if hasattr(self.model, "means_"):
            self.model.means_ = self.model.means_[perm]

        cov_type = getattr(self.model, "covariance_type", self.cfg.covariance_type)
        raw_covars = getattr(self.model, "_covars_", None)
        if isinstance(raw_covars, np.ndarray) and raw_covars.shape:
            if cov_type in {"diag", "spherical", "full"} and raw_covars.shape[0] == n:
                self.model._covars_ = raw_covars[perm]
            return

        covars = getattr(self.model, "covars_", None)
        if isinstance(covars, np.ndarray) and covars.shape and covars.shape[0] == n:
            self.model.covars_ = covars[perm]

    @staticmethod
    def _build_canonical_semantic_diagnostics(
        semantic_diag: SemanticOrderingDiagnostics,
        n_states: int,
    ) -> dict[str, object]:
        canonical_order = [int(x) for x in semantic_diag.canonical_order]
        inverse = HMMEngine._inverse_permutation(canonical_order)

        semantic_to_canonical: dict[str, int] = {}
        for semantic_name, old_state in semantic_diag.semantic_to_original_state.items():
            semantic_to_canonical[semantic_name] = int(inverse[int(old_state)])

        canonical_profiles: list[dict[str, object]] = []
        for row in semantic_diag.state_profiles:
            old_state = int(row.get("state_id", -1))
            new_state = int(inverse[old_state]) if old_state >= 0 else -1
            canonical_profiles.append({**row, "original_state_id": old_state, "state_id": new_state})

        canonical_profiles = sorted(canonical_profiles, key=lambda r: int(r.get("state_id", 10**6)))

        payload = {
            "canonical_order_from_original": canonical_order,
            "semantic_order_matches_topology_before_reorder": bool(
                semantic_diag.semantic_order_matches_topology
            ),
            "semantic_to_state": semantic_to_canonical,
            "semantic_confidence": {
                k: float(v) for k, v in semantic_diag.semantic_confidence.items()
            },
            "state_profiles": canonical_profiles,
            "warnings": list(semantic_diag.warnings),
            "canonical_order": list(range(n_states)),
        }
        return payload

    def _build_canonical_state_mapping(self) -> dict[str, object]:
        n_states = int(getattr(self.model, "n_components", self.cfg.n_hidden_states))
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
            "state_order_used_for_transitions": canonical_ids,
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

    def canonical_state_mapping(self) -> dict[str, object]:
        if self.last_canonical_state_mapping:
            return self.last_canonical_state_mapping
        self.last_canonical_state_mapping = self._build_canonical_state_mapping()
        return self.last_canonical_state_mapping

    def fit(self, features: pd.DataFrame, sequence_ids: pd.Series | None = None) -> float:
        features = features.copy().reset_index(drop=True)
        self.post_scale_weights = self._build_post_scale_weights(list(features.columns))
        batch = encode_observations(
            features=features,
            scaler=self.scaler,
            fit_scaler=True,
            sequence_ids=sequence_ids,
            post_scale_weights=self.post_scale_weights,
        )
        self.feature_columns = batch.feature_columns

        self._initialize_semantic_priors(x=batch.values, features=features)

        trainer = HMMTrainer(
            self.model,
            topology_mode=self.cfg.topology_mode,
            min_forward_transition=float(getattr(self.cfg, "min_forward_transition", 0.0)),
        )
        result = trainer.fit(batch.values, lengths=batch.lengths)

        train_states_original = self.model.predict(
            batch.values,
            lengths=batch.lengths if batch.lengths else None,
        )
        semantic_diag = derive_semantic_ordering(
            features=features,
            decoded_states=train_states_original,
            n_states=self.cfg.n_hidden_states,
        )

        canonical_order = semantic_diag.canonical_order
        identity_order = list(range(self.cfg.n_hidden_states))
        if getattr(self.cfg, "canonical_reorder_enabled", True) and canonical_order != identity_order:
            self._reorder_model_states(canonical_order)
            trainer.apply_topology_constraints()

        self.last_training_result = TrainingResult(
            log_likelihood=result.log_likelihood,
            transition_matrix=self.model.transmat_.copy(),
            converged=result.converged,
            n_iterations=result.n_iterations,
            last_delta=result.last_delta,
        )

        self.state_definition = semantic_diag.canonical_state_definition(self.cfg.n_hidden_states)
        self.last_semantic_diagnostics = self._build_canonical_semantic_diagnostics(
            semantic_diag=semantic_diag,
            n_states=self.cfg.n_hidden_states,
        )
        self.last_canonical_state_mapping = self._build_canonical_state_mapping()

        warning_text = "; ".join(self.last_semantic_diagnostics.get("warnings", []))
        logger.info(
            "HMM fitted. log_likelihood=%.4f, n_sequences=%d, converged=%s, n_iter=%d, last_delta=%s",
            result.log_likelihood,
            len(batch.lengths) if batch.lengths else 1,
            result.converged,
            result.n_iterations,
            None if result.last_delta is None else round(float(result.last_delta), 6),
        )
        if warning_text:
            logger.info("Semantic diagnostics: %s", warning_text)

        return result.log_likelihood

    def predict(self, features: pd.DataFrame, sequence_ids: pd.Series | None = None) -> HMMPrediction:
        features = self._align_features(features)
        batch = encode_observations(
            features=features,
            scaler=self.scaler,
            fit_scaler=False,
            sequence_ids=sequence_ids,
            post_scale_weights=self.post_scale_weights,
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
            "post_scale_weights": self.post_scale_weights,
            "semantic_diagnostics": self.last_semantic_diagnostics,
            "canonical_state_mapping": self.last_canonical_state_mapping,
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
        obj.post_scale_weights = payload.get("post_scale_weights")
        obj.last_semantic_diagnostics = payload.get("semantic_diagnostics")
        obj.last_canonical_state_mapping = payload.get("canonical_state_mapping")
        if obj.last_canonical_state_mapping is None:
            obj.last_canonical_state_mapping = obj._build_canonical_state_mapping()
        return obj
