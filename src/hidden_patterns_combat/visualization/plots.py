from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .charts import plot_hidden_state_sequence, plot_state_probability_profile


def plot_hidden_states(states: pd.Series, out_path: str | Path) -> None:
    frame = pd.DataFrame({"hidden_state": states.values})
    plot_hidden_state_sequence(frame, out_path=out_path, state_col="hidden_state")


def plot_state_probabilities(probabilities: np.ndarray, out_path: str | Path) -> None:
    frame = pd.DataFrame(probabilities).add_prefix("p_state_")
    plot_state_probability_profile(frame, out_path=out_path, prob_prefix="p_state_")
