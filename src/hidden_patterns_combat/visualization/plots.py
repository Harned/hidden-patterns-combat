from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _get_pyplot():
    # Keep matplotlib cache inside the project/writable space.
    mpl_dir = Path("artifacts/.mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_hidden_states(states: pd.Series, out_path: str | Path) -> None:
    plt = _get_pyplot()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 3))
    plt.step(np.arange(len(states)), states.values, where="mid")
    plt.title("Most Probable Hidden-State Sequence")
    plt.xlabel("Episode index")
    plt.ylabel("Hidden state")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_state_probabilities(probabilities: np.ndarray, out_path: str | Path) -> None:
    plt = _get_pyplot()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    for i in range(probabilities.shape[1]):
        plt.plot(probabilities[:, i], label=f"State {i}")
    plt.title("Hidden-State Probability Profile")
    plt.xlabel("Episode index")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
