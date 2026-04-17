from __future__ import annotations

import inspect
import json
from pathlib import Path

from openpyxl import Workbook
import pandas as pd
import pytest

from hidden_patterns_combat.app.inverse_diagnostic_cycle import run_inverse_diagnostic_cycle


def _header_schema() -> list[tuple[str, str, str]]:
    schema: list[tuple[str, str, str]] = [
        ("Идентификация", "Общее", "ФИО борца"),
        ("Идентификация", "Общее", "№ эпизода"),
        ("Идентификация", "Общее", "Время эпизода, с."),
        ("Идентификация", "Общее", "Время паузы, с."),
        ("Баллы", "Общее", "Баллы"),
    ]

    for i in range(1, 13):
        schema.append(("S1", "Правосторонняя стойка", f"ПС_{i:02d}"))
    for i in range(1, 13):
        schema.append(("S1", "Левосторонняя стойка", f"ЛС_{i:02d}"))

    for i in range(1, 16):
        schema.append(("S2", "Захваты", f"ЗХ_{i:02d}"))
    for i in range(1, 4):
        schema.append(("S2", "Хваты", f"ХВ_{i:02d}"))
    for i in range(1, 4):
        schema.append(("S2", "Обхваты", f"ОБ_{i:02d}"))
    for i in range(1, 5):
        schema.append(("S2", "Прихваты", f"ПР_{i:02d}"))
    for i in range(1, 5):
        schema.append(("S2", "Упоры", f"УП_{i:02d}"))

    for i in range(1, 6):
        schema.append(("S3", "ВУП", f"ВУП_{i:02d}"))

    schema.extend(
        [
            ("O", "ЗАП", "Броски Руками"),
            ("O", "ЗАП", "Броски Ногами"),
            ("O", "ЗАП", "Броски Туловищем"),
            ("O", "ЗАП", "Удержание"),
            ("O", "ЗАП", "Болевой на руку"),
            ("O", "ЗАП", "Болевой на ногу"),
        ]
    )
    return schema


def _blank_episode_row() -> dict[str, float | str]:
    row = {leaf: 0 for _, _, leaf in _header_schema()}
    row["ФИО борца"] = ""
    row["№ эпизода"] = ""
    row["Время эпизода, с."] = 0
    row["Время паузы, с."] = 0
    row["Баллы"] = 0
    return row


def _make_episode(
    *,
    athlete: str,
    episode_id: int | str,
    score: float,
    o_class: str | None,
    high_counts: bool = False,
) -> dict[str, float | str]:
    row = _blank_episode_row()
    row["ФИО борца"] = athlete
    row["№ эпизода"] = episode_id
    row["Время эпизода, с."] = 12
    row["Время паузы, с."] = 4
    row["Баллы"] = score

    strength = 3 if high_counts else 1
    row["ПС_01"] = strength
    row["ЛС_01"] = 1

    row["ЗХ_01"] = strength
    row["ХВ_01"] = strength
    row["ОБ_01"] = 1
    row["ПР_01"] = 1
    row["УП_01"] = 1

    row["ВУП_01"] = strength

    if o_class == "O1":
        row["Броски Руками"] = 1
    elif o_class == "O2":
        row["Броски Ногами"] = 1
    elif o_class == "O3":
        row["Броски Туловищем"] = 1
    elif o_class == "O4":
        row["Удержание"] = 1
    elif o_class == "O5":
        row["Болевой на руку"] = 1
    elif o_class == "O6":
        row["Болевой на ногу"] = 1

    return row


def _write_multilevel_workbook(path: Path, rows: list[dict[str, float | str]]) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Эпизоды"

    schema = _header_schema()
    ws.append([g for g, _, _ in schema])
    ws.append([s for _, s, _ in schema])
    ws.append([l for _, _, l in schema])

    leaves = [leaf for _, _, leaf in schema]
    for row in rows:
        ws.append([row.get(leaf, 0) for leaf in leaves])

    wb.save(path)
    return path


def test_signature_stays_backward_compatible() -> None:
    sig = inspect.signature(run_inverse_diagnostic_cycle)
    for name in [
        "input_path",
        "output_dir",
        "sheet_names",
        "header_depth",
        "parser_mode",
        "force_matrix_parser",
        "retrain",
        "model_path",
        "reset_outputs",
        "n_states",
        "topology_mode",
        "generate_plots",
        "verbose",
        "cleanup_mode",
        "isolated_run",
        "run_id",
    ]:
        assert name in sig.parameters


def test_viterbi_path_with_o1_confident(tmp_path: Path) -> None:
    excel_path = _write_multilevel_workbook(
        tmp_path / "synthetic_o1.xlsx",
        rows=[_make_episode(athlete="Иванов", episode_id=1, score=4, o_class="O1", high_counts=True)],
    )

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=tmp_path / "artifacts_o1",
        n_states=4,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    viterbi = pd.read_csv(Path(result.final_output_dir) / "diagnostics" / "per_episode_viterbi.csv")
    assert len(viterbi) == 4
    assert viterbi["step_hidden_state"].tolist() == ["S1", "S2", "S3", "O"]
    assert (viterbi["confidence"] >= 0.9).all()

    analysis = pd.read_csv(result.episode_analysis_path)
    assert set(analysis["observed_zap_class"].astype(str).unique()) == {"O1"}


def test_episode_without_o_finishes_with_o0_or_s3(tmp_path: Path) -> None:
    excel_path = _write_multilevel_workbook(
        tmp_path / "synthetic_o0.xlsx",
        rows=[_make_episode(athlete="Петров", episode_id=1, score=0, o_class=None)],
    )

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=tmp_path / "artifacts_o0",
        n_states=4,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    viterbi = pd.read_csv(Path(result.final_output_dir) / "diagnostics" / "per_episode_viterbi.csv")
    assert len(viterbi) in {3, 4}
    assert viterbi.iloc[-1]["step_hidden_state"] in {"S3", "O"}

    analysis = pd.read_csv(result.episode_analysis_path)
    assert set(analysis["observed_zap_class"].astype(str).unique()) == {"O0"}


def test_degenerate_dataset_sets_status_degenerate(tmp_path: Path) -> None:
    rows = [
        _make_episode(athlete="Сидоров", episode_id=i + 1, score=0, o_class=None)
        for i in range(12)
    ]
    excel_path = _write_multilevel_workbook(tmp_path / "degenerate.xlsx", rows=rows)

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=tmp_path / "artifacts_degenerate",
        n_states=4,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    run_summary = json.loads(Path(result.run_summary_path).read_text(encoding="utf-8"))
    quality = json.loads(Path(result.quality_diagnostics_path).read_text(encoding="utf-8"))

    assert run_summary["status"] == "degenerate"
    assert quality["status"] == "degenerate"
    assert quality["tripwires_triggered"]
    assert quality["dominant_state_share"] <= 0.9
    assert quality["self_transition_share"] >= 0.0


def test_loader_regression_multilevel_ffill_drop_totals_and_binarize(tmp_path: Path) -> None:
    rows = [
        _make_episode(athlete="Кузнецов", episode_id=1, score=2, o_class="O4", high_counts=True),
        _make_episode(athlete="", episode_id=2, score=0, o_class=None, high_counts=True),
        _blank_episode_row(),
        {
            **_make_episode(athlete="Итог", episode_id="Итог", score=0, o_class=None),
            "ПС_01": 9,
            "ЗХ_01": 9,
        },
    ]

    excel_path = _write_multilevel_workbook(tmp_path / "loader_regression.xlsx", rows=rows)

    result = run_inverse_diagnostic_cycle(
        input_path=excel_path,
        output_dir=tmp_path / "artifacts_loader",
        n_states=4,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    canonical = pd.read_csv(result.canonical_episode_table_path)
    assert len(canonical) == 2
    assert (canonical["athlete_name"].astype(str).str.strip() != "").all()
    assert not canonical["episode_id"].astype(str).str.contains("итог", case=False, na=False).any()

    features = pd.read_csv(Path(result.final_output_dir) / "features" / "episode_features.csv")
    assert "s1_ps_b01_count" in features.columns
    assert "s1_ps_b01_bin" in features.columns
    assert float(features["s1_ps_b01_count"].max()) > 1.0
    assert int(features["s1_ps_b01_bin"].max()) == 1


def test_smoke_real_episodes_if_present(tmp_path: Path) -> None:
    real_path = Path("data/raw/episodes.xlsx")
    if not real_path.exists():
        pytest.skip("data/raw/episodes.xlsx is not available in this environment")

    result = run_inverse_diagnostic_cycle(
        input_path=real_path,
        output_dir=tmp_path / "artifacts_real_smoke",
        n_states=4,
        retrain=True,
        generate_plots=False,
        verbose=False,
    )

    quality = json.loads(Path(result.quality_diagnostics_path).read_text(encoding="utf-8"))
    assert float(quality.get("dominant_state_share", 1.0)) < 0.9
