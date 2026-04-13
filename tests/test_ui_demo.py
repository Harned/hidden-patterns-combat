from pathlib import Path

import pandas as pd

from hidden_patterns_combat.ui.mvp_cli import _choose_plot, _extract_episode_insight


def test_extract_episode_insight_from_analysis_csv(tmp_path: Path):
    csv_path = tmp_path / "episode_analysis.csv"
    pd.DataFrame(
        {
            "episode_id": ["e1", "e2"],
            "hidden_state": [1, 2],
            "hidden_state_name": ["state_1", "state_2"],
            "maneuver_right_code": [3, 4],
            "kfv_code": [1, 2],
            "vup_code": [0, 1],
            "outcome_actions_code": [1, 0],
            "observed_result": [2, 0],
        }
    ).to_csv(csv_path, index=False)

    insight = _extract_episode_insight(csv_path, 1)
    assert insight.episode_id == "e2"
    assert insight.hidden_state == "state_2"
    assert insight.hidden_state_id == 2
    assert "kfv_code" in insight.key_features


def test_choose_plot_prefers_hidden_state_sequence(tmp_path: Path):
    p1 = tmp_path / "hidden_state_sequence.png"
    p2 = tmp_path / "state_probability_profile.png"
    p2.write_bytes(b"x")
    p1.write_bytes(b"x")
    selected = _choose_plot(tmp_path)
    assert selected == p1
