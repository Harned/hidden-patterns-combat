"""Тесты HMM-ветки (TASK_SPEC_004).

Охватывают:
* guard'ы блокируют HMM на тонких данных;
* на плотной фикстуре HMM сходится и даёт status=hmm_ready;
* инварианты (states=маневрирование/КФВ/ВУП, observations=ЗАП,
  fighter_style отсутствует, HMM-поля появляются только при hmm_ready);
* воспроизводимость seed.
"""

from __future__ import annotations

from pathlib import Path

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.schema import AnalysisStatus

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_guards_block_hmm_on_thin_data(thin_excel: Path) -> None:
    cfg = preflight_mapping(thin_excel)
    result = analyze_source(thin_excel, AnalyzeConfig(column_mapping=cfg))

    assert result.status != AnalysisStatus.HMM_READY
    assert result.hmm is None
    codes = {w.code for w in result.warnings}
    assert "hmm.guards_failed" in codes


def test_hmm_disabled_via_config(dense_hmm_ready_excel: Path) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    result = analyze_source(
        dense_hmm_ready_excel,
        AnalyzeConfig(column_mapping=cfg, enable_hmm=False),
    )
    assert result.status == AnalysisStatus.BASELINE_ONLY
    assert result.hmm is None
    codes = {w.code for w in result.warnings}
    assert "hmm.disabled" in codes


# ---------------------------------------------------------------------------
# Успешная HMM
# ---------------------------------------------------------------------------


def test_hmm_ready_on_dense_synthetic(dense_hmm_ready_excel: Path) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    result = analyze_source(
        dense_hmm_ready_excel,
        AnalyzeConfig(column_mapping=cfg, hmm_seed=42),
    )

    assert result.status == AnalysisStatus.HMM_READY
    assert result.hmm is not None

    params = result.hmm.parameters
    assert params.n_states == 3
    assert params.state_labels == ["маневрирование", "КФВ", "ВУП"]
    # observation labels содержат как минимум один из каналов ЗАП.
    assert any(
        label in {"Удержание", "На руку", "На ногу", "ЗАП-Р"}
        for label in params.observation_labels
    )

    # Viterbi пути непустые, все состояния из доменного списка.
    assert result.hmm.trajectories
    for tr in result.hmm.trajectories:
        for s in tr.state_path:
            assert s in {"маневрирование", "КФВ", "ВУП"}

    # Sanity-check явно помечен пройденным.
    assert result.hmm.sanity["transition_dominance_ok"] is True

    # Chart-id'ы HMM присутствуют.
    chart_ids = {c.id for c in result.charts}
    assert "hmm_transition_matrix" in chart_ids
    assert "hmm_state_distribution" in chart_ids


# ---------------------------------------------------------------------------
# Инварианты и воспроизводимость
# ---------------------------------------------------------------------------


def test_hmm_fields_absent_without_hmm_ready(thin_excel: Path) -> None:
    cfg = preflight_mapping(thin_excel)
    result = analyze_source(thin_excel, AnalyzeConfig(column_mapping=cfg))
    assert result.hmm is None
    # state_labels в результате тоже не должны появляться как payload.
    dumped = result.model_dump()
    assert dumped.get("hmm") is None


def test_fighter_style_is_never_used(dense_hmm_ready_excel: Path) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    result = analyze_source(dense_hmm_ready_excel, AnalyzeConfig(column_mapping=cfg))
    dumped = result.model_dump_json()
    assert "fighter_style" not in dumped


def test_reproducibility_same_seed(dense_hmm_ready_excel: Path) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    a = analyze_source(
        dense_hmm_ready_excel,
        AnalyzeConfig(column_mapping=cfg, hmm_seed=7),
    )
    b = analyze_source(
        dense_hmm_ready_excel,
        AnalyzeConfig(column_mapping=cfg, hmm_seed=7),
    )
    assert a.status == AnalysisStatus.HMM_READY
    assert b.status == AnalysisStatus.HMM_READY
    assert a.hmm is not None and b.hmm is not None
    assert a.hmm.parameters.log_likelihood == b.hmm.parameters.log_likelihood


def test_different_seeds_may_differ(dense_hmm_ready_excel: Path) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    a = analyze_source(
        dense_hmm_ready_excel,
        AnalyzeConfig(column_mapping=cfg, hmm_seed=1),
    )
    b = analyze_source(
        dense_hmm_ready_excel,
        AnalyzeConfig(column_mapping=cfg, hmm_seed=999),
    )
    # Хотя бы один из двух должен быть hmm_ready — это защита от
    # случая, когда модель не сошлась при нестандартном seed.
    assert AnalysisStatus.HMM_READY in {a.status, b.status}
