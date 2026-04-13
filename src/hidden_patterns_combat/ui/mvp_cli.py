from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.pipeline import CombatHMMPipeline
from hidden_patterns_combat.preprocessing import run_preprocessing


@dataclass
class EpisodeInsight:
    episode_index: int
    episode_id: str
    hidden_state: str
    hidden_state_id: int
    latent_state_message: str
    key_features: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DemoUIResult:
    preprocessing: dict[str, object]
    analysis: dict[str, object]
    episode_insight: dict[str, object]
    visualization_path: str | None
    interpretation_text: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _choose_plot(analysis_dir: Path) -> Path | None:
    candidates = [
        "hidden_state_sequence.png",
        "state_probability_profile.png",
        "scenario_success_frequencies.png",
        "transition_distribution.png",
        "athlete_comparative_profile.png",
    ]
    for name in candidates:
        p = analysis_dir / name
        if p.exists():
            return p
    return None


def _extract_episode_insight(analysis_csv: Path, episode_index: int) -> EpisodeInsight:
    df = pd.read_csv(analysis_csv)
    if df.empty:
        raise ValueError("Analysis output is empty.")

    idx = max(0, min(episode_index, len(df) - 1))
    row = df.iloc[idx]

    feature_cols = [
        "maneuver_right_code",
        "maneuver_left_code",
        "grips_code",
        "holds_code",
        "bodylocks_code",
        "underhooks_code",
        "posts_code",
        "kfv_code",
        "vup_code",
        "outcome_actions_code",
        "observed_result",
    ]
    key = {c: float(row[c]) for c in feature_cols if c in df.columns}

    state_name = str(row.get("hidden_state_name", row.get("hidden_state", "unknown")))
    state_id = int(row.get("hidden_state", -1))
    episode_id = str(row.get("episode_id", idx))

    return EpisodeInsight(
        episode_index=idx,
        episode_id=episode_id,
        hidden_state=state_name,
        hidden_state_id=state_id,
        latent_state_message=f"Наиболее вероятное латентное состояние эпизода: {state_name}",
        key_features=key,
    )


def run_demo_workflow(
    excel_path: str,
    sheet: str | None = None,
    model_path: str = "artifacts/hmm_model.pkl",
    preprocess_output_dir: str = "data/processed/preprocessing",
    analysis_output_dir: str = "artifacts/analysis",
    episode_index: int = 0,
    n_states: int = 3,
    retrain: bool = False,
) -> DemoUIResult:
    """End-user MVP flow for CLI/notebook demo.

    Steps:
    1) preprocessing;
    2) train model when needed or requested;
    3) analysis/decode;
    4) derive interpretable episode-level insight.
    """
    preprocess_report = run_preprocessing(
        excel_path=excel_path,
        sheet_selector=sheet,
        output_dir=preprocess_output_dir,
    ).to_dict()

    cfg = PipelineConfig()
    cfg.model.n_hidden_states = n_states
    pipeline = CombatHMMPipeline(cfg)

    model_file = Path(model_path)
    if retrain or (not model_file.exists()):
        pipeline.train(excel_path=excel_path, model_out=model_path, sheet=sheet)

    analysis_report = pipeline.analyze(
        excel_path=excel_path,
        model_path=model_path,
        output_dir=analysis_output_dir,
        sheet=sheet,
    )

    analysis_dir = Path(analysis_output_dir)
    episode_insight = _extract_episode_insight(analysis_dir / "episode_analysis.csv", episode_index)

    interpretation_path = analysis_dir / "interpretation.txt"
    interpretation_text = interpretation_path.read_text(encoding="utf-8") if interpretation_path.exists() else ""
    brief_text = "\n".join(interpretation_text.strip().splitlines()[:3])

    selected_plot = _choose_plot(analysis_dir)

    return DemoUIResult(
        preprocessing=preprocess_report,
        analysis=analysis_report,
        episode_insight=episode_insight.to_dict(),
        visualization_path=str(selected_plot) if selected_plot else None,
        interpretation_text=brief_text,
    )
