from __future__ import annotations

import os
from pathlib import Path


def get_pyplot():
    """Return configured matplotlib.pyplot for headless report generation."""
    mpl_dir = Path("artifacts/.mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def ensure_output_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def default_state_colors(n: int) -> list[str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]
