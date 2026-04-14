from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hidden_patterns_combat.config import ModelConfig, PipelineConfig
from hidden_patterns_combat.modeling.hmm_pipeline import HMMEngine
from hidden_patterns_combat.modeling.state_definition import build_semantic_state_definition
from hidden_patterns_combat.pipeline import CombatHMMPipeline


HAS_HMMLEARN = importlib.util.find_spec("hmmlearn") is not None


def test_build_semantic_state_definition_maps_s1_s2_s3():
    features = pd.DataFrame(
        {
            "maneuver_right_code": [5, 4, 0, 0, 0, 0],
            "maneuver_left_code": [3, 2, 0, 0, 0, 0],
            "kfv_code": [0, 0, 6, 7, 0, 0],
            "grips_code": [0, 0, 1, 1, 0, 0],
            "vup_code": [0, 0, 0, 0, 4, 5],
            "duration": [1, 1, 1, 1, 1, 1],
            "pause": [0, 0, 0, 0, 0, 0],
        }
    )
    states = np.array([0, 0, 1, 1, 2, 2])

    definition = build_semantic_state_definition(features, states, n_states=3)
    assert set(definition.state_names()) == {"S1", "S2", "S3"}


def test_build_semantic_state_definition_keeps_neutral_state_when_signal_is_weak():
    features = pd.DataFrame(
        {
            "maneuver_right_code": [4, 4, 0, 0, 0, 0],
            "maneuver_left_code": [3, 3, 0, 0, 0, 0],
            "kfv_code": [0, 0, 6, 6, 1, 1],
            "grips_code": [0, 0, 2, 2, 1, 1],
            "vup_code": [0, 0, 0, 0, 1, 1],
        }
    )
    states = np.array([0, 0, 1, 1, 2, 2])

    definition = build_semantic_state_definition(features, states, n_states=3)
    names = set(definition.state_names())
    assert "S1" in names
    assert "S2" in names
    assert "S3" not in names
    assert any(name.startswith("state_") for name in names)


@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn missing")
def test_hmm_engine_fit_assigns_and_persists_semantic_names(tmp_path: Path):
    cfg = ModelConfig(n_hidden_states=3, n_iter=40, random_state=7, topology_mode="left_to_right")
    engine = HMMEngine(cfg)

    features = pd.DataFrame(
        {
            "maneuver_right_code": [4] * 15 + [0] * 30,
            "maneuver_left_code": [2] * 15 + [0] * 30,
            "grips_code": [0] * 15 + [2] * 15 + [0] * 15,
            "holds_code": [0] * 15 + [1] * 15 + [0] * 15,
            "bodylocks_code": [0] * 15 + [1] * 15 + [0] * 15,
            "underhooks_code": [0] * 15 + [1] * 15 + [0] * 15,
            "posts_code": [0] * 15 + [1] * 15 + [0] * 15,
            "kfv_code": [0] * 15 + [5] * 15 + [0] * 15,
            "vup_code": [0] * 30 + [4] * 15,
            "duration": [1] * 45,
            "pause": [0] * 45,
        }
    )
    sequence_ids = pd.Series(["s"] * len(features))

    engine.fit(features, sequence_ids=sequence_ids)
    assert set(engine.state_definition.state_names()) == {"S1", "S2", "S3"}

    model_path = tmp_path / "hmm.pkl"
    engine.save(model_path)
    loaded = HMMEngine.load(model_path)
    assert loaded.state_definition.state_names() == engine.state_definition.state_names()


@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn missing")
def test_pipeline_analyze_exports_semantic_hidden_state_names(demo_excel_path: Path, tmp_path: Path):
    cfg = PipelineConfig()
    cfg.model.n_hidden_states = 3
    cfg.model.topology_mode = "left_to_right"
    cfg.model.n_iter = 30
    pipeline = CombatHMMPipeline(cfg)

    model_path = tmp_path / "model.pkl"
    pipeline.train(demo_excel_path, model_path, sheet=None, parser_mode="auto")
    out = pipeline.analyze(demo_excel_path, model_path, tmp_path / "analysis", sheet=None, parser_mode="auto")

    analysis_df = pd.read_csv(out["analysis_csv"])
    assert "hidden_state_name" in analysis_df.columns
    unique = set(analysis_df["hidden_state_name"].dropna().astype(str).unique().tolist())
    assert unique
    assert all(name.startswith("S") or name.startswith("state_") for name in unique)
