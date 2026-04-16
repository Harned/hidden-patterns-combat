from __future__ import annotations

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


def test_cleanup_artifacts_only_removes_pipeline_artifacts_but_keeps_user_files(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_cleanup.xlsx")
    output_dir = tmp_path / "inverse_cleanup_artifacts"

    stale_cleaned = output_dir / "cleaned" / "stale.csv"
    stale_diag = output_dir / "diagnostics" / "stale.json"
    user_root_file = output_dir / "user_notes.txt"
    model_cache_file = output_dir / "models" / "manual_cache.txt"

    stale_cleaned.parent.mkdir(parents=True, exist_ok=True)
    stale_diag.parent.mkdir(parents=True, exist_ok=True)
    model_cache_file.parent.mkdir(parents=True, exist_ok=True)
    stale_cleaned.write_text("stale\n", encoding="utf-8")
    stale_diag.write_text("{\"stale\": true}\n", encoding="utf-8")
    user_root_file.write_text("keep me\n", encoding="utf-8")
    model_cache_file.write_text("cache\n", encoding="utf-8")

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        cleanup_mode="artifacts_only",
        generate_plots=False,
        verbose=False,
    )

    assert result.cleanup_mode == "artifacts_only"
    assert stale_cleaned.exists() is False
    assert stale_diag.exists() is False
    assert user_root_file.exists() is True
    assert model_cache_file.exists() is True


def test_fixed_output_uses_safe_cleanup_by_default(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_cleanup_default.xlsx")
    output_dir = tmp_path / "inverse_cleanup_default_artifacts"

    stale_file = output_dir / "features" / "stale_features.csv"
    user_file = output_dir / "custom_notes.txt"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("old\n", encoding="utf-8")
    user_file.write_text("keep\n", encoding="utf-8")

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_dir,
        generate_plots=False,
        verbose=False,
    )

    assert result.cleanup_mode == "artifacts_only"
    assert stale_file.exists() is False
    assert user_file.exists() is True
