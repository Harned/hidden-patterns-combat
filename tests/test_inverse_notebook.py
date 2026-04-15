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

    run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    artifacts = load_inverse_artifacts(output_dir)
    assert not artifacts.episode_analysis.empty
    assert not artifacts.state_profile.empty
    assert isinstance(artifacts.observation_audit, dict)
    assert isinstance(artifacts.metadata_extraction_summary, dict)
    assert isinstance(artifacts.sequence_audit, dict)
    assert isinstance(artifacts.model_health_summary, dict)
    assert not artifacts.observation_mapping_crosstab.empty
    assert artifacts.raw_finish_signal_summary.shape[0] >= 0
    assert artifacts.sequence_length_distribution.shape[0] >= 0
    assert artifacts.suspicious_sequences.shape[0] >= 0
