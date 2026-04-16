from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
import json
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
    ws.append(["ФИО борца", "Последовательность", "Технико-тактический эпизод", "Весовая категория", "Баллы", "m1", "k1", "v1", "h1", "a1", "l1"])

    ws.append(["A", "bout_1", 1, "79", 0, 1, 0, 0, 0, 0, 0])
    ws.append(["A", "bout_1", 2, "79", 1, 1, 1, 0, 0, 0, 0])
    ws.append(["A", "bout_1", 3, "79", 2, 0, 1, 0, 1, 0, 0])
    ws.append(["A", "bout_2", 1, "79", 4, 0, 0, 1, 0, 1, 0])
    ws.append(["A", "bout_2", 2, "79", 0, 1, 0, 0, 0, 0, 0])
    ws.append(["B", "bout_3", 1, "88", 3, 0, 0, 1, 0, 0, 0])
    ws.append(["B", "bout_3", 2, "88", 0, 1, 0, 0, 0, 0, 0])

    wb.save(path)
    return path


def test_inverse_pipeline_end_to_end_with_quality_columns(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_demo.xlsx")
    output_dir = tmp_path / "inverse_artifacts"

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        retrain=True,
        generate_plots=True,
        verbose=False,
    )

    assert Path(result.episode_analysis_path).exists()
    assert Path(result.state_profile_path).exists()
    assert Path(result.quality_diagnostics_path).exists()
    assert Path(result.report_path).exists()
    assert Path(result.run_manifest_path).exists()
    assert result.final_output_dir == str(output_dir)

    analysis = pd.read_csv(result.episode_analysis_path)
    assert "observed_zap_class" in analysis.columns
    assert "observation_resolution_type" in analysis.columns
    assert "observation_confidence_label" in analysis.columns
    assert "observation_quality_flag" in analysis.columns
    assert "sequence_id" in analysis.columns
    assert "hidden_state" in analysis.columns
    assert "confidence" in analysis.columns

    observed_values = set(analysis["observed_zap_class"].astype(str).unique().tolist())
    assert "no_score" in observed_values
    assert "unknown" in observed_values

    assert isinstance(result.observed_layer_summary, dict)
    assert isinstance(result.sequence_quality_summary, dict)
    assert isinstance(result.recommendation, str)
    assert result.recommendation.strip() != ""

    report_text = Path(result.report_path).read_text(encoding="utf-8")
    assert "## 2) Observed layer quality" in report_text
    assert "## 5) Sequence segmentation quality" in report_text
    assert "## 9) Limitations" in report_text

    quality = json.loads(Path(result.quality_diagnostics_path).read_text(encoding="utf-8"))
    assert "topology_compliance" in quality
    assert "state_anchor_alignment" in quality
    assert "finish_proximity" in quality
    assert "semantic_stability" in quality
    assert "train_composition" in quality
