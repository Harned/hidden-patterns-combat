from __future__ import annotations

import pandas as pd

from hidden_patterns_combat.diagnostics.metadata_audit import build_metadata_extraction_summary


def test_metadata_extraction_summary_marks_missing_and_non_informative_fields() -> None:
    canonical = pd.DataFrame(
        {
            "athlete_name": ["A", "A", "B"],
            "athlete_id": ["id_a", "id_a", "id_b"],
            "sheet_name": ["S", "S", "S"],
            "weight_class": ["", "", ""],
            "episode_id": ["1", "2", "1"],
            "episode_time_sec": [0.0, 0.0, 0.0],
            "pause_time_sec": [0.0, 0.0, 0.0],
            "score": [1.0, 0.0, 2.0],
            "sequence_id": ["s1", "s1", "s2"],
        }
    )
    summary = build_metadata_extraction_summary(canonical_df=canonical, extraction_info={}).summary

    assert summary["weight_class_informative"] is False
    assert summary["episode_time_informative"] is False
    assert "weight_class" in summary["missing_metadata_fields"]
    assert any("Weight class" in warning for warning in summary["warnings"])
