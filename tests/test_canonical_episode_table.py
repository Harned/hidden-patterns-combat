from __future__ import annotations

import pandas as pd

from hidden_patterns_combat.preprocessing.canonical_episode_table import build_canonical_episode_table


def test_canonical_episode_table_excludes_total_rows_from_training() -> None:
    cleaned = pd.DataFrame(
        {
            "metadata__athlete_name": ["Иванов И.И.", "Итог"],
            "metadata__sheet": ["Общее", "Общее"],
            "metadata__episode_id": ["1", "2"],
            "outcomes__score": [1, 5],
        }
    )
    observations = pd.DataFrame(
        {
            "observed_zap_class": ["zap_r", "zap_t"],
            "observed_zap_source_columns": ["[]", "[]"],
            "observation_quality_flag": ["ok_score_rule", "ok_score_rule"],
            "observation_resolution_type": ["inferred_from_score", "inferred_from_score"],
            "observation_confidence_label": ["medium", "medium"],
            "mapping_version": ["observation_mapping_v1", "observation_mapping_v1"],
        }
    )
    hidden = pd.DataFrame(
        {
            "maneuver_right_code": [1, 0],
            "maneuver_left_code": [0, 0],
            "grips_code": [1, 0],
            "holds_code": [0, 0],
            "bodylocks_code": [0, 0],
            "underhooks_code": [0, 0],
            "posts_code": [0, 0],
            "vup_code": [0, 0],
        }
    )

    canonical = build_canonical_episode_table(cleaned, observations, hidden).canonical_table

    assert bool(canonical.loc[0, "is_total_row"]) is False
    assert bool(canonical.loc[0, "is_train_eligible"]) is True
    assert bool(canonical.loc[1, "is_total_row"]) is True
    assert bool(canonical.loc[1, "is_train_eligible"]) is False


def test_canonical_episode_table_sequence_id_is_deterministic() -> None:
    cleaned = pd.DataFrame(
        {
            "metadata__athlete_name": ["A", "A", "A", "A"],
            "metadata__sheet": ["S", "S", "S", "S"],
            "metadata__episode_id": ["1", "2", "1", "2"],
            "outcomes__score": [1, 1, 1, 1],
        }
    )
    observations = pd.DataFrame(
        {
            "observed_zap_class": ["zap_r", "zap_r", "zap_r", "zap_r"],
            "observed_zap_source_columns": ["[]", "[]", "[]", "[]"],
            "observation_quality_flag": ["ok_score_rule"] * 4,
            "observation_resolution_type": ["inferred_from_score"] * 4,
            "observation_confidence_label": ["medium"] * 4,
            "mapping_version": ["observation_mapping_v1"] * 4,
        }
    )

    canonical_a = build_canonical_episode_table(cleaned, observations).canonical_table
    canonical_b = build_canonical_episode_table(cleaned, observations).canonical_table

    assert canonical_a["sequence_id"].tolist() == canonical_b["sequence_id"].tolist()
    assert canonical_a["sequence_id"].nunique() >= 2


def test_canonical_episode_table_keeps_traceability_fields() -> None:
    cleaned = pd.DataFrame(
        {
            "metadata__athlete_name": ["Петров П.П."],
            "metadata__sheet": ["48"],
            "metadata__episode_id": ["A-3"],
            "outcomes__score": [0],
        }
    )
    observations = pd.DataFrame(
        {
            "observed_zap_class": ["no_score"],
            "observed_zap_source_columns": ["[]"],
            "observation_quality_flag": ["ok_no_score_rule"],
            "observation_resolution_type": ["no_score_rule"],
            "observation_confidence_label": ["high"],
            "mapping_version": ["observation_mapping_v1"],
        }
    )

    canonical = build_canonical_episode_table(cleaned, observations).canonical_table

    assert canonical.loc[0, "source_row_index"] == 0
    assert canonical.loc[0, "source_record_id"] == "48::0"
    assert canonical.loc[0, "sheet_name"] == "48"


def test_canonical_episode_table_marks_low_quality_sequences() -> None:
    cleaned = pd.DataFrame(
        {
            "metadata__athlete_name": ["A", "A"],
            "metadata__sheet": ["Sheet1", "Sheet1"],
            "metadata__episode_id": ["ep_a", "ep_b"],
            "outcomes__score": [1, 1],
        }
    )
    observations = pd.DataFrame(
        {
            "observed_zap_class": ["zap_r", "zap_r"],
            "observed_zap_source_columns": ["[]", "[]"],
            "observation_quality_flag": ["ok_score_rule", "ok_score_rule"],
            "observation_resolution_type": ["inferred_from_score", "inferred_from_score"],
            "observation_confidence_label": ["medium", "medium"],
            "mapping_version": ["observation_mapping_v1", "observation_mapping_v1"],
        }
    )

    canonical = build_canonical_episode_table(cleaned, observations).canonical_table

    assert set(canonical["sequence_quality_flag"].astype(str).tolist()) == {"low"}
    assert canonical["is_train_eligible"].eq(False).all()
