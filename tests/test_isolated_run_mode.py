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


def test_isolated_run_mode_creates_unique_output_dirs(tmp_path: Path) -> None:
    excel_path = _make_inverse_excel(tmp_path / "inverse_isolated.xlsx")
    output_base = tmp_path / "inverse_runs"

    result_1 = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_base,
        isolated_run=True,
        generate_plots=False,
        verbose=False,
    )
    result_2 = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=output_base,
        isolated_run=True,
        generate_plots=False,
        verbose=False,
    )

    assert result_1.final_output_dir != result_2.final_output_dir
    assert result_1.run_id != result_2.run_id
    assert Path(result_1.final_output_dir).exists()
    assert Path(result_2.final_output_dir).exists()
    assert Path(result_1.final_output_dir).parent == output_base
    assert Path(result_2.final_output_dir).parent == output_base
