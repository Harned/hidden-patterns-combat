"""Тесты детализированной 7-state HMM (TASK_SPEC_005)."""

from __future__ import annotations

from pathlib import Path

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.hmm import (
    STATE_LABELS_BASIC,
    STATE_LABELS_DETAILED,
    VARIANT_BASIC,
    VARIANT_DETAILED,
)
from hpc_algo.schema import AnalysisStatus


def _analyze(path: Path, **overrides) -> object:
    cfg = preflight_mapping(path)
    return analyze_source(path, AnalyzeConfig(column_mapping=cfg, **overrides))


# ---------------------------------------------------------------------------
# Thin data → detailed blocked
# ---------------------------------------------------------------------------


def test_detailed_blocked_on_thin_data(dense_hmm_ready_excel: Path) -> None:
    """40 эпизодов — detailed не должен пройти min_episodes_detailed=120."""

    result = _analyze(dense_hmm_ready_excel, hmm_mode="detailed")
    assert result.status == AnalysisStatus.BASELINE_ONLY
    assert result.hmm is None
    codes = {w.code for w in result.warnings}
    assert "hmm.fit_failed" in codes


def test_auto_falls_back_to_basic_on_thin_data(
    dense_hmm_ready_excel: Path,
) -> None:
    result = _analyze(dense_hmm_ready_excel, hmm_mode="auto")
    assert result.status == AnalysisStatus.HMM_READY
    assert result.hmm is not None
    assert result.hmm.parameters.variant == VARIANT_BASIC
    assert result.hmm.parameters.state_labels == list(STATE_LABELS_BASIC)


# ---------------------------------------------------------------------------
# Very dense data → detailed succeeds
# ---------------------------------------------------------------------------


def test_detailed_runs_on_very_dense(very_dense_excel: Path) -> None:
    result = _analyze(very_dense_excel, hmm_mode="detailed")
    assert result.status == AnalysisStatus.HMM_READY
    assert result.hmm is not None
    assert result.hmm.parameters.variant == VARIANT_DETAILED
    assert result.hmm.parameters.state_labels == list(STATE_LABELS_DETAILED)
    assert result.hmm.parameters.n_states == 7
    assert result.hmm.parameters.bic is not None
    assert result.hmm.sanity["enough_states_used"] is True


def test_auto_prefers_basic_unless_bic_wins(very_dense_excel: Path) -> None:
    """BIC-гейт: даже на плотной фикстуре 7-state не всегда выигрывает BIC.

    Инвариант: auto возвращает одну из двух валидных моделей, но НИКОГДА
    не возвращает detailed, если её BIC хуже basic.
    """

    result = _analyze(very_dense_excel, hmm_mode="auto")
    assert result.status == AnalysisStatus.HMM_READY
    assert result.hmm is not None
    variant = result.hmm.parameters.variant
    assert variant in {VARIANT_BASIC, VARIANT_DETAILED}

    # Если auto выбрал detailed — его BIC должен быть строго лучше basic.
    if variant == VARIANT_DETAILED:
        # Повторно обучим basic и сравним BIC.
        basic = _analyze(very_dense_excel, hmm_mode="basic")
        assert basic.hmm is not None
        assert (
            result.hmm.parameters.bic is not None
            and basic.hmm.parameters.bic is not None
            and result.hmm.parameters.bic < basic.hmm.parameters.bic
        )


# ---------------------------------------------------------------------------
# Инварианты предметной области
# ---------------------------------------------------------------------------


def test_domain_state_names_detailed(very_dense_excel: Path) -> None:
    result = _analyze(very_dense_excel, hmm_mode="detailed")
    assert result.hmm is not None
    for tr in result.hmm.trajectories:
        for s in tr.state_path:
            assert s in set(STATE_LABELS_DETAILED)


def test_no_fighter_style_anywhere(very_dense_excel: Path) -> None:
    result = _analyze(very_dense_excel, hmm_mode="detailed")
    dumped = result.model_dump_json()
    assert "fighter_style" not in dumped


def test_off_mode_disables_hmm(very_dense_excel: Path) -> None:
    result = _analyze(very_dense_excel, hmm_mode="off")
    assert result.status == AnalysisStatus.BASELINE_ONLY
    assert result.hmm is None
