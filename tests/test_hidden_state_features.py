from __future__ import annotations

import pandas as pd

from hidden_patterns_combat.features.hidden_state_features import build_hidden_state_feature_layer


def test_hidden_state_feature_layer_builds_anchor_and_duration_columns() -> None:
    canonical = pd.DataFrame(
        {
            "maneuver_right_code": [6.0, 0.5, 0.2],
            "maneuver_left_code": [4.0, 0.2, 0.1],
            "kfv_capture_code": [0.1, 3.0, 0.3],
            "kfv_grip_code": [0.1, 2.5, 0.2],
            "kfv_wrap_code": [0.1, 1.8, 0.2],
            "kfv_hook_code": [0.1, 1.2, 0.2],
            "kfv_post_code": [0.1, 0.9, 0.1],
            "vup_code": [0.1, 0.2, 4.5],
            "episode_time_sec": [10.0, 22.0, 44.0],
            "pause_time_sec": [8.0, 4.0, 2.0],
            "observed_zap_class": ["no_score", "zap_n", "zap_t"],
            "observation_resolution_type": ["no_score_rule", "inferred_from_score", "direct_finish_signal"],
            "observation_confidence_label": ["high", "medium", "high"],
            "sequence_quality_flag": ["medium", "medium", "high"],
            "sequence_resolution_type": ["surrogate", "surrogate", "explicit"],
            "is_train_eligible": [True, True, True],
        }
    )

    layer = build_hidden_state_feature_layer(canonical).hidden_state_features

    for col in ("duration_bin", "pause_bin", "anchor_s1", "anchor_s2", "anchor_s3", "train_weight"):
        assert col in layer.columns

    assert set(layer["duration_bin"].astype(int).tolist()).issubset({0, 1, 2})
    assert int(layer.loc[0, "anchor_s1"] > layer.loc[1, "anchor_s1"]) == 1
    assert int(layer.loc[1, "anchor_s2"] > layer.loc[0, "anchor_s2"]) == 1
    assert int(layer.loc[2, "anchor_s3"] > layer.loc[1, "anchor_s3"]) == 1


def test_hidden_state_feature_layer_downweights_no_score_and_unknown_rows() -> None:
    canonical = pd.DataFrame(
        {
            "maneuver_right_code": [1, 1, 1, 1],
            "maneuver_left_code": [0, 0, 0, 0],
            "kfv_capture_code": [0, 0, 0, 0],
            "kfv_grip_code": [0, 0, 0, 0],
            "kfv_wrap_code": [0, 0, 0, 0],
            "kfv_hook_code": [0, 0, 0, 0],
            "kfv_post_code": [0, 0, 0, 0],
            "vup_code": [0, 0, 0, 0],
            "episode_time_sec": [12, 18, 24, 30],
            "pause_time_sec": [4, 4, 4, 4],
            "observed_zap_class": ["no_score", "unknown", "zap_t", "zap_n"],
            "observation_resolution_type": [
                "no_score_rule",
                "ambiguous",
                "direct_finish_signal",
                "inferred_from_score",
            ],
            "observation_confidence_label": ["high", "low", "high", "medium"],
            "sequence_quality_flag": ["medium", "medium", "high", "high"],
            "sequence_resolution_type": ["surrogate", "surrogate", "explicit", "explicit"],
            "is_train_eligible": [True, False, True, True],
        }
    )

    layer = build_hidden_state_feature_layer(canonical).hidden_state_features
    weights = layer["train_weight"].astype(float).tolist()

    assert weights[2] > weights[3] > weights[0]
    assert weights[1] == 0.0


def test_hidden_state_feature_layer_anchor_components_are_row_normalized_and_contrasting() -> None:
    canonical = pd.DataFrame(
        {
            "sequence_id": ["seq_1", "seq_1", "seq_1"],
            "source_row_index": [0, 1, 2],
            "maneuver_right_code": [10.0, 1.0, 0.2],
            "maneuver_left_code": [8.0, 0.5, 0.1],
            "kfv_capture_code": [0.2, 6.0, 0.3],
            "kfv_grip_code": [0.2, 4.0, 0.3],
            "kfv_wrap_code": [0.2, 3.0, 0.2],
            "kfv_hook_code": [0.1, 2.0, 0.2],
            "kfv_post_code": [0.1, 1.5, 0.1],
            "vup_code": [0.1, 0.8, 5.0],
            "episode_time_sec": [8.0, 24.0, 42.0],
            "pause_time_sec": [10.0, 6.0, 2.0],
            "observed_zap_class": ["zap_r", "zap_n", "zap_t"],
            "observation_resolution_type": [
                "direct_finish_signal",
                "inferred_from_score",
                "direct_finish_signal",
            ],
            "observation_confidence_label": ["high", "medium", "high"],
            "sequence_quality_flag": ["high", "high", "high"],
            "is_train_eligible": [True, True, True],
        }
    )

    layer = build_hidden_state_feature_layer(canonical).hidden_state_features
    row_sums = (layer["anchor_s1"] + layer["anchor_s2"] + layer["anchor_s3"]).round(6)

    assert (row_sums == 1.0).all()
    assert layer.loc[0, "anchor_s1"] > layer.loc[0, "anchor_s2"]
    assert layer.loc[1, "anchor_s2"] > layer.loc[1, "anchor_s1"]
    assert layer.loc[2, "anchor_s3"] > layer.loc[2, "anchor_s2"]
