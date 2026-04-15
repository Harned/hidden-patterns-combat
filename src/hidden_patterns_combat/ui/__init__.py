"""User-facing MVP interaction layer (CLI + notebook)."""

from .inverse_notebook import (
    InverseNotebookArtifacts,
    display_inverse_plots,
    display_inverse_report,
    load_inverse_artifacts,
)
from .mvp_cli import DemoUIResult, EpisodeInsight, run_demo_workflow

__all__ = [
    "EpisodeInsight",
    "DemoUIResult",
    "run_demo_workflow",
    "InverseNotebookArtifacts",
    "load_inverse_artifacts",
    "display_inverse_report",
    "display_inverse_plots",
]
