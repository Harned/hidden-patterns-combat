from pathlib import Path

import pandas as pd

from hidden_patterns_combat.visualization.charts import (
    _state_probability_legend_labels,
    _transition_plot_rows,
    create_analysis_charts,
)


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


def test_state_probability_legend_labels_use_canonical_mapping():
    df = pd.DataFrame(
        {
            "hidden_state": [0, 1, 2],
            "hidden_state_name": ["S1", "state_0", "state_1"],
            "p_state_0": [0.8, 0.2, 0.1],
            "p_state_1": [0.1, 0.7, 0.2],
            "p_state_2": [0.1, 0.1, 0.7],
        }
    )

    labels = _state_probability_legend_labels(
        df,
        prob_cols=["p_state_0", "p_state_1", "p_state_2"],
        state_name_map={0: "S1", 1: "state_0", 2: "state_1"},
    )
    assert labels == {
        "p_state_0": "S1",
        "p_state_1": "state_0",
        "p_state_2": "state_1",
    }


def test_transition_plot_rows_use_provided_transition_summary():
    df = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1"],
            "hidden_state_name": ["S1", "S1", "S1"],
        }
    )
    rows = _transition_plot_rows(
        df,
        transition_summary=[
            {"from_name": "S1", "to_name": "state_0", "count": 7},
            {"from_name": "state_0", "to_name": "state_1", "count": 3},
        ],
        top_k=10,
    )
    assert rows == [("S1 -> state_0", 7), ("state_0 -> state_1", 3)]


def test_transition_plot_rows_use_contiguous_sequence_boundaries():
    df = pd.DataFrame(
        {
            "sequence_id": ["a", "a", "b", "a"],
            "hidden_state_name": ["S1", "state_0", "state_1", "S1"],
        }
    )
    rows = _transition_plot_rows(df, top_k=10)
    as_dict = dict(rows)
    assert as_dict.get("S1 -> state_0", 0) == 1
    assert as_dict.get("state_0 -> state_1", 0) == 0
