from __future__ import annotations

import numpy as np
import pandas as pd

from hidden_patterns_combat.config import ModelConfig
from hidden_patterns_combat.modeling.inverse_hmm import InverseDiagnosticHMM


def _obs_classes() -> list[str]:
    return [
        "zap_r",
        "zap_n",
        "zap_t",
        "hold",
        "arm_submission",
        "leg_submission",
        "no_score",
        "unknown",
    ]


def test_inverse_hmm_enforces_strict_left_to_right_constraints() -> None:
    cfg = ModelConfig(n_hidden_states=3, topology_mode="left_to_right")
    model = InverseDiagnosticHMM(cfg=cfg, observation_classes=_obs_classes())

    trans = np.array(
        [
            [0.95, 0.01, 0.04],
            [0.45, 0.50, 0.05],
            [0.25, 0.15, 0.60],
        ],
        dtype=float,
    )
    constrained = model._enforce_topology(trans)

    assert constrained[0, 2] == 0.0
    assert constrained[1, 0] == 0.0
    assert constrained[2, 0] == 0.0
    assert constrained[2, 1] == 0.0
    assert constrained[0, 1] >= cfg.min_forward_transition - 1e-9
    assert constrained[1, 2] >= cfg.min_forward_transition - 1e-9
    assert constrained[0, 0] <= cfg.max_self_transition + 1e-9
    assert constrained[0, 0] >= cfg.min_self_transition - 1e-9
    assert constrained[1, 1] >= cfg.min_self_transition - 1e-9


def test_inverse_hmm_semantic_assignment_is_deterministic_with_anchor_signals() -> None:
    cfg = ModelConfig(n_hidden_states=3, topology_mode="left_to_right")
    model = InverseDiagnosticHMM(cfg=cfg, observation_classes=_obs_classes())

    features = pd.DataFrame(
        {
            "anchor_s1": [0.95, 0.92, 0.90, 0.10, 0.15, 0.12, 0.05, 0.04, 0.03],
            "anchor_s2": [0.05, 0.08, 0.10, 0.92, 0.96, 0.94, 0.10, 0.08, 0.06],
            "anchor_s3": [0.05, 0.05, 0.08, 0.10, 0.15, 0.12, 0.95, 0.96, 0.94],
            "duration_bin": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "pause_bin": [2, 2, 2, 1, 1, 1, 0, 0, 0],
        }
    )
    decoded_states = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1], dtype=int)
    observed = [
        "no_score",
        "no_score",
        "zap_r",
        "zap_n",
        "hold",
        "zap_n",
        "zap_t",
        "arm_submission",
        "leg_submission",
    ]

    semantic = model._derive_inverse_semantic_ordering(
        semantic_features=features,
        decoded_states=decoded_states,
        observed_labels=observed,
        lengths=[len(observed)],
    )

    assert semantic.semantic_to_original_state == {"S1": 2, "S2": 0, "S3": 1}
    assert semantic.canonical_order == [2, 0, 1]


def test_inverse_hmm_downweights_low_information_rows_for_training() -> None:
    cfg = ModelConfig(n_hidden_states=3, topology_mode="left_to_right")
    model = InverseDiagnosticHMM(cfg=cfg, observation_classes=_obs_classes())
    features = pd.DataFrame(
        {
            "train_weight": [0.80, 0.80, 0.80, 0.80],
            "observation_resolution_type": [
                "direct_finish_signal",
                "inferred_from_score",
                "no_score_rule",
                "ambiguous",
            ],
            "observation_confidence_label": ["high", "medium", "high", "low"],
        }
    )
    observed = ["zap_t", "zap_n", "no_score", "unknown"]

    weights = model._row_train_weights(features, observed_labels=observed)

    assert weights[0] > weights[1] > weights[2] > weights[3]
    assert float(weights[3]) < 0.01


def test_inverse_hmm_semantic_confidence_penalizes_anchor_conflicts() -> None:
    cfg = ModelConfig(n_hidden_states=3, topology_mode="left_to_right")
    model = InverseDiagnosticHMM(cfg=cfg, observation_classes=_obs_classes())

    features = pd.DataFrame(
        {
            "anchor_s1": [0.95, 0.94, 0.93, 0.60, 0.58, 0.57, 0.06, 0.05, 0.04],
            "anchor_s2": [0.03, 0.04, 0.05, 0.55, 0.54, 0.53, 0.08, 0.07, 0.06],
            "anchor_s3": [0.02, 0.02, 0.02, 0.25, 0.28, 0.30, 0.96, 0.95, 0.94],
            "duration_bin": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "pause_bin": [2, 2, 2, 1, 1, 1, 0, 0, 0],
        }
    )
    decoded_states = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=int)
    observed = ["no_score", "zap_r", "zap_n", "zap_n", "zap_n", "hold", "zap_t", "hold", "leg_submission"]

    semantic = model._derive_inverse_semantic_ordering(
        semantic_features=features,
        decoded_states=decoded_states,
        observed_labels=observed,
        lengths=[len(observed)],
    )

    assert semantic.semantic_to_original_state.get("S2") == 1
    assert semantic.semantic_confidence.get("S2", 1.0) < semantic.semantic_confidence.get("S1", 0.0)
    assert any("conflicts with dominant anchor" in warning for warning in semantic.warnings)


def test_inverse_hmm_predict_uses_anchor_priors_when_hidden_features_provided() -> None:
    cfg = ModelConfig(
        n_hidden_states=3,
        topology_mode="left_to_right",
        inverse_inference_anchor_strength=1.2,
    )
    model = InverseDiagnosticHMM(cfg=cfg, observation_classes=_obs_classes())
    model.startprob_ = np.array([1.0, 0.0, 0.0], dtype=float)
    model.transmat_ = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    model.emissionprob_ = np.array(
        [
            [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0],
            [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0],
        ],
        dtype=float,
    )
    model.last_semantic_diagnostics = {
        "canonical_order_from_original": [0, 1, 2],
        "semantic_to_state": {"S1": 0, "S2": 1, "S3": 2},
        "semantic_confidence": {"S1": 0.8, "S2": 0.8, "S3": 0.8},
        "state_profiles": [],
        "warnings": [],
        "semantic_order_matches_topology_before_reorder": True,
    }
    model.last_canonical_state_mapping = model.canonical_state_mapping()

    observed = pd.Series(["no_score"] * 12)
    sequence_ids = pd.Series(["seq_1"] * len(observed))
    features = pd.DataFrame(
        {
            "sequence_progress": np.linspace(0.0, 1.0, len(observed)),
            "anchor_s1": [0.95] * 4 + [0.10] * 4 + [0.05] * 4,
            "anchor_s2": [0.03] * 4 + [0.85] * 4 + [0.10] * 4,
            "anchor_s3": [0.02] * 4 + [0.05] * 4 + [0.85] * 4,
            "duration_bin": [0] * 4 + [1] * 4 + [2] * 4,
            "pause_bin": [2] * 4 + [1] * 4 + [0] * 4,
            "observation_resolution_type": ["no_score_rule"] * len(observed),
            "observation_confidence_label": ["high"] * len(observed),
        }
    )

    plain = model.predict(observed, sequence_ids=sequence_ids)
    anchored = model.predict(observed, sequence_ids=sequence_ids, hidden_state_features=features)

    assert len(set(plain.states.tolist())) <= 2
    assert len(set(anchored.states.tolist())) >= 2
    assert anchored.states[-1] == 2
