from __future__ import annotations

from pathlib import Path

from hpc_algo.api import analysis_summary, analyze_source, is_honest_baseline
from hpc_algo.schema import AnalysisStatus, HiddenGroup


def test_analyze_source_returns_baseline_for_zap_file(zap_only_excel: Path) -> None:
    result = analyze_source(zap_only_excel)

    assert is_honest_baseline(result)
    assert result.status == AnalysisStatus.BASELINE_ONLY
    assert HiddenGroup.ZAP in result.detected_columns.detected_groups
    # Baseline-распределение по уверенной ЗАП-колонке должно существовать.
    assert result.basic_statistics.zap_value_counts, (
        "Ожидается хотя бы одно распределение ЗАП для уверенной колонки"
    )


def test_analyze_source_flags_missing_mapping(opaque_excel: Path) -> None:
    result = analyze_source(opaque_excel)

    assert result.status == AnalysisStatus.NEEDS_COLUMN_MAPPING
    # Не должно быть никаких ЗАП-распределений.
    assert result.basic_statistics.zap_value_counts == {}
    # Обязательное предупреждение.
    codes = {w.code for w in result.warnings}
    assert "status.needs_column_mapping" in codes
    assert "detection.no_strong_zap" in codes


def test_analyze_source_never_fabricates_hmm_fields(full_structure_excel: Path) -> None:
    """Ключевая проверка честности: HMM-поля в MVP отсутствуют."""

    result = analyze_source(full_structure_excel)
    dumped = result.model_dump()

    for forbidden in ("viterbi_path", "hidden_states", "gamma", "transition_matrix"):
        assert forbidden not in dumped, (
            f"В MVP не должно быть поля '{forbidden}' в AnalysisResult"
        )
    assert result.status != AnalysisStatus.HMM_READY


def test_analyze_source_failed_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "no_such.xlsx"
    result = analyze_source(missing)

    assert result.status == AnalysisStatus.FAILED
    assert result.errors
    assert result.data_audit.sheets == []


def test_analysis_summary_structure(zap_only_excel: Path) -> None:
    result = analyze_source(zap_only_excel)
    summary = analysis_summary(result)

    assert summary["status"] == AnalysisStatus.BASELINE_ONLY.value
    assert summary["filename"].endswith(".xlsx")
    assert summary["total_rows"] == 4
    assert any(c["score"] >= 0.7 for c in summary["zap_candidates"])


def test_observations_are_zap_invariant(full_structure_excel: Path) -> None:
    """Domain invariant: наблюдения — ЗАП, а не действия спортсмена."""

    result = analyze_source(full_structure_excel)

    # Все уверенные кандидаты на роль observation должны относиться к группе ЗАП.
    strong_zap = [
        c for c in result.detected_columns.candidates
        if c.group == HiddenGroup.ZAP and c.score >= 0.7
    ]
    assert strong_zap

    # fighter_style не должен появляться в detected_columns / отчёте.
    report_lower = result.report.lower()
    assert "fighter_style" not in report_lower
