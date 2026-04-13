from pathlib import Path

import pandas as pd

from hidden_patterns_combat.visualization.charts import create_analysis_charts


def test_create_analysis_charts_smoke(tmp_path: Path):
    df = pd.DataFrame(
        {
            "episode_id": ["1", "2", "3", "4"],
            "athlete_name": ["A", "A", "B", "B"],
            "sequence_id": ["s1", "s1", "s2", "s2"],
            "hidden_state": [0, 1, 1, 2],
            "hidden_state_name": ["S1", "S2", "S2", "S3"],
            "observed_result": [1, 0, 2, 0],
            "p_state_0": [0.8, 0.1, 0.2, 0.1],
            "p_state_1": [0.1, 0.8, 0.7, 0.2],
            "p_state_2": [0.1, 0.1, 0.1, 0.7],
        }
    )

    outputs = create_analysis_charts(df, tmp_path)
    assert "hidden_state_sequence" in outputs
    assert "state_probability_profile" in outputs
    assert "scenario_success_frequencies" in outputs
    assert "transition_distribution" in outputs

    for _, path in outputs.items():
        assert Path(path).exists()


def test_success_failure_plot_handles_single_class(tmp_path: Path):
    df = pd.DataFrame(
        {
            "hidden_state": [0, 0, 1],
            "hidden_state_name": ["S1", "S1", "S2"],
            "observed_result": [0, 0, 0],  # only unsuccessful class
            "p_state_0": [0.8, 0.7, 0.2],
            "p_state_1": [0.2, 0.3, 0.8],
        }
    )

    outputs = create_analysis_charts(df, tmp_path)
    assert Path(outputs["scenario_success_frequencies"]).exists()
