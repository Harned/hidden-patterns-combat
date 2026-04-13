"""User-facing MVP interaction layer (CLI + notebook)."""

from .mvp_cli import DemoUIResult, EpisodeInsight, run_demo_workflow

__all__ = ["EpisodeInsight", "DemoUIResult", "run_demo_workflow"]
