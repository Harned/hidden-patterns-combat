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
            "mapping_version": ["observation_mapping_v1"],
        }
    )

    canonical = build_canonical_episode_table(cleaned, observations).canonical_table

    assert canonical.loc[0, "source_row_index"] == 0
    assert canonical.loc[0, "source_record_id"] == "48::0"
    assert canonical.loc[0, "sheet_name"] == "48"
