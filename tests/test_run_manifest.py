from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook

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


def test_run_manifest_created_and_contains_core_fields(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_manifest.xlsx")
    output_dir = tmp_path / "inverse_manifest_artifacts"

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        generate_plots=False,
        verbose=False,
    )

    manifest_path = Path(result.run_manifest_path)
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload.get("status") == "completed"
    assert payload.get("pipeline_mode") == "inverse-diagnostic"
    assert payload.get("run_id") == result.run_id
    assert payload.get("started_at")
    assert payload.get("finished_at")
    assert payload.get("input_path") == str(excel_path)
    assert payload.get("input_file_name") == excel_path.name
    assert payload.get("input_file_hash")
    assert payload.get("output_dir") == str(output_dir)
    assert payload.get("n_states") == 3
    assert payload.get("topology_mode") == "left_to_right"
    assert payload.get("mapping_version")
    assert payload.get("number_of_episodes") is not None
    assert payload.get("number_of_train_eligible_episodes") is not None
    assert payload.get("number_of_sequences") is not None

    expected = payload.get("expected_artifact_files", [])
    created = payload.get("created_artifact_files", [])
    assert isinstance(expected, list) and expected
    assert isinstance(created, list) and created
    assert "run_manifest.json" in expected
    assert "run_manifest.json" in created
