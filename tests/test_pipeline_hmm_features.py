from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.pipeline import CombatHMMPipeline


HAS_HMMLEARN = importlib.util.find_spec("hmmlearn") is not None


@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn missing")
def test_pipeline_train_excludes_outcome_features_from_hmm_input(demo_excel_path: Path, tmp_path: Path):
    cfg = PipelineConfig()
    cfg.model.n_hidden_states = 2
    cfg.model.n_iter = 30
    pipeline = CombatHMMPipeline(cfg)

    report = pipeline.train(
        excel_path=demo_excel_path,
        model_out=tmp_path / "model.pkl",
        sheet="Общее",
        parser_mode="auto",
    )

    feature_names = report["features"]
    assert "outcome_actions_code" not in feature_names
    assert "observed_result" not in feature_names
