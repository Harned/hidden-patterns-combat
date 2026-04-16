from __future__ import annotations

import json
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


def test_loader_warns_when_manifest_missing(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_loader_missing_manifest.xlsx")
    output_dir = tmp_path / "inverse_loader_missing_manifest_artifacts"

    run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        generate_plots=False,
        verbose=False,
    )

    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    artifacts = load_inverse_artifacts(output_dir)
    assert any("run_manifest_missing_or_invalid" in warning for warning in artifacts.loader_warnings)


def test_loader_warns_when_manifest_and_artifacts_are_inconsistent(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_loader_stale.xlsx")
    output_dir = tmp_path / "inverse_loader_stale_artifacts"

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        generate_plots=False,
        verbose=False,
    )

    manifest_path = output_dir / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = list(payload.get("expected_artifact_files", []) or [])
    expected.append("diagnostics/missing_from_disk.csv")
    payload["expected_artifact_files"] = expected
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    unexpected = output_dir / "diagnostics" / "stale_leftover.csv"
    unexpected.write_text("stale\n", encoding="utf-8")

    artifacts = load_inverse_artifacts(
        output_dir,
        expected_run_id=f"{result.run_id}_other",
        expected_run_fingerprint=result.run_fingerprint,
    )
    assert any("missing_expected_artifact" in warning for warning in artifacts.loader_warnings)
    assert any("unexpected_artifact_not_in_manifest" in warning for warning in artifacts.loader_warnings)
    assert any("run_id_mismatch" in warning for warning in artifacts.loader_warnings)
