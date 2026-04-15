from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
import pandas as pd

from hidden_patterns_combat.app.inverse_diagnostic_cycle import run_inverse_diagnostic_cycle


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


def test_inverse_diagnostic_cycle_end_to_end(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_demo.xlsx")
    output_dir = tmp_path / "inverse_artifacts"

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        retrain=True,
        generate_plots=True,
        verbose=False,
    )

    assert Path(result.cleaned_data_path).exists()
    assert Path(result.canonical_episode_table_path).exists()
    assert Path(result.observed_sequence_path).exists()
    assert Path(result.hidden_feature_layer_path).exists()
    assert Path(result.episode_analysis_path).exists()
    assert Path(result.state_profile_path).exists()
    assert Path(result.report_path).exists()

    for required_dir in ["cleaned", "features", "diagnostics", "plots", "reports"]:
        assert (output_dir / required_dir).exists()
    assert (output_dir / "plots" / "hidden_state_sequence.png").exists()
    assert (output_dir / "plots" / "state_probability_profile.png").exists()

    analysis = pd.read_csv(result.episode_analysis_path)
    assert "observed_zap_class" in analysis.columns
    assert "hidden_state" in analysis.columns
    assert "hidden_state_name" in analysis.columns
    assert "confidence" in analysis.columns
    assert any(col.startswith("p_state_") for col in analysis.columns)

    observed_values = set(analysis["observed_zap_class"].astype(str).unique().tolist())
    assert "no_score" in observed_values
    assert "unknown" in observed_values

    assert isinstance(result.recommendation, str)
    assert result.recommendation.strip() != ""
