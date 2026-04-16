from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from hidden_patterns_combat.app.inverse_diagnostic_cycle import run_inverse_diagnostic_cycle
from hidden_patterns_combat.ui.inverse_notebook import load_inverse_artifacts


def _make_inverse_excel(path: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Общее"

    ws.append(
        [
            None,
            None,
            None,
            "Стойка и маневрирование самбиста (основные в эпизоде)",
            "Контакты Физического Взаимодействия (захваты, обхваты, прихваты, хваты, упоры)",
            "Выведение соперника из устойчивого положения (при выполнении n или n1)",
            "Удержание",
            "Болевой на руку",
            "Болевой на ногу",
        ]
    )
    ws.append(["ФИО борца", "Технико-тактический эпизод", "Баллы", "m1", "k1", "v1", "h1", "a1", "l1"])
    ws.append(["A", 1, 0, 1, 0, 0, 0, 0, 0])
    ws.append(["A", 2, 1, 1, 1, 0, 0, 0, 0])
    ws.append(["A", 3, 2, 0, 1, 0, 1, 0, 0])
    ws.append(["A", 4, 4, 0, 0, 1, 0, 1, 0])
    ws.append(["B", 1, 2, 0, 1, 0, 0, 0, 0])
    ws.append(["B", 2, 3, 0, 0, 1, 0, 0, 0])
    ws.append(["B", 3, 0, 1, 0, 0, 0, 0, 0])
    wb.save(path)
    return path


def test_inverse_notebook_loader_reads_extended_diagnostics(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_demo.xlsx")
    output_dir = tmp_path / "inverse_artifacts"

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    artifacts = load_inverse_artifacts(
        output_dir,
        expected_run_id=result.run_id,
        expected_run_fingerprint=result.run_fingerprint,
    )
    assert isinstance(artifacts.run_summary, dict)
    assert "semantic_assignment_quality" in artifacts.run_summary
    assert artifacts.run_manifest.get("run_id") == result.run_id
    assert artifacts.missing_expected_artifacts == []
    assert not artifacts.episode_analysis.empty
    assert not artifacts.state_profile.empty
    assert isinstance(artifacts.observation_audit, dict)
    assert isinstance(artifacts.metadata_extraction_summary, dict)
    assert isinstance(artifacts.sequence_audit, dict)
    assert isinstance(artifacts.model_health_summary, dict)
    assert not artifacts.observation_mapping_crosstab.empty
    assert artifacts.raw_finish_signal_summary.shape[0] >= 0
    assert artifacts.unsupported_finish_values.shape[0] >= 0
    assert artifacts.metadata_field_coverage.shape[0] >= 0
    assert artifacts.sequence_length_distribution.shape[0] >= 0
    assert artifacts.suspicious_sequences.shape[0] >= 0
    assert isinstance(artifacts.artifact_status, type(artifacts.episode_analysis))
    assert "artifact_name" in artifacts.artifact_status.columns
    assert "status" in artifacts.artifact_status.columns


def test_inverse_notebook_loader_reports_missing_artifacts(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_demo_missing.xlsx")
    output_dir = tmp_path / "inverse_artifacts_missing"

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    missing_sequence = output_dir / "diagnostics" / "sequence_audit.json"
    if missing_sequence.exists():
        missing_sequence.unlink()
    missing_run_summary = output_dir / "diagnostics" / "run_summary.json"
    if missing_run_summary.exists():
        missing_run_summary.unlink()

    artifacts = load_inverse_artifacts(
        output_dir,
        expected_run_id=result.run_id,
        expected_run_fingerprint=result.run_fingerprint,
    )
    status = artifacts.artifact_status.set_index("artifact_name")["status"].to_dict()
    assert status.get("sequence_audit") == "missing"
    assert status.get("run_summary") == "missing"
    assert any("sequence_audit" in warning for warning in artifacts.loader_warnings)
    assert any("run_summary" in warning for warning in artifacts.loader_warnings)
    assert any("missing_expected_artifact" in warning for warning in artifacts.loader_warnings)
