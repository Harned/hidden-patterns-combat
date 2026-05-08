"""End-to-end тесты оркестратора :mod:`hpc_algo.build_aggregate` (TS_012).

Используем фикстуру ``markov_two_bouts_excel`` (Иванов / Петров) и
вручную пишем ``finalists.yaml``, в котором делаем их призёрами в
одной фиктивной весовой категории. Это синтетика для проверки
пайплайна, не реальный ЧР-2025.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hpc_algo.build_aggregate import build_aggregate_models
from hpc_algo.build_individual import _slugify
from hpc_algo.episode_split import read_episodes_sheet


def _state_groups_yaml(excel_path: Path, yaml_path: Path) -> None:
    df = read_episodes_sheet(excel_path, sheet="Общее")
    cols = list(df.columns)
    payload = {
        "version": "1",
        "sheet": "Общее",
        "mode": "single",
        "priority": [
            "technical_action",
            "off_balance",
            "grip",
            "manoeuvring",
            "pause",
        ],
        "states": {
            "manoeuvring": [c for c in cols if "Стойка" in c],
            "grip": [c for c in cols if "КФВ" in c],
            "off_balance": [c for c in cols if "ВУП" in c],
            "technical_action": [
                c for c in cols if ("Завершающие" in c or "Болевой" in c)
            ],
        },
    }
    yaml_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _finalists_yaml(yaml_path: Path) -> None:
    yaml_path.write_text(
        """
version: 1
weight_classes:
  "48":
    1: "Иванов"
    2: "Петров"
""",
        encoding="utf-8",
    )


def test_build_aggregate_end_to_end(
    markov_two_bouts_excel: Path, tmp_path: Path
) -> None:
    sg_path = tmp_path / "state_groups.yaml"
    _state_groups_yaml(markov_two_bouts_excel, sg_path)
    fin_path = tmp_path / "finalists.yaml"
    _finalists_yaml(fin_path)
    out_dir = tmp_path / "reports_aggregate"

    summary = build_aggregate_models(
        excel_path=markov_two_bouts_excel,
        state_groups_path=sg_path,
        output_dir=out_dir,
        finalists_path=fin_path,
        sheet="Общее",
        alpha=0.5,
    )

    assert summary.weight_classes_total == 1
    assert summary.weight_classes_rendered == 1
    assert summary.rendered_weight_classes == ["48"]
    assert summary.per_class_member_counts == {"48": 2}

    html_path = out_dir / f"{_slugify('48')}.html"
    assert html_path.exists()
    text = html_path.read_text(encoding="utf-8")
    assert "Агрегатная Marков-цепь" in text
    assert "Категория" in text
    assert "Иванов" in text
    assert "Петров" in text
    assert "Ранжирование финалистов" in text

    assert (out_dir / "index.html").exists()
    assert (out_dir / "summary.json").exists()
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["alpha"] == 0.5


def test_build_aggregate_excludes_non_medalist(
    markov_two_bouts_excel: Path, tmp_path: Path
) -> None:
    """place=4 → не попадает в агрегат, warning aggregate.non_medalist_dropped."""

    sg_path = tmp_path / "state_groups.yaml"
    _state_groups_yaml(markov_two_bouts_excel, sg_path)

    fin_path = tmp_path / "finalists.yaml"
    fin_path.write_text(
        """
weight_classes:
  "48":
    1: "Иванов"
    4: "Петров"
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "reports_filtered"
    summary = build_aggregate_models(
        excel_path=markov_two_bouts_excel,
        state_groups_path=sg_path,
        output_dir=out_dir,
        finalists_path=fin_path,
        sheet="Общее",
    )
    # Только Иванов как призёр.
    assert summary.per_class_member_counts == {"48": 1}
    codes = {w.code for w in summary.finalists_warnings}
    assert "aggregate.non_medalist_dropped" in codes


def test_build_aggregate_empty_finalists_skips_classes(
    markov_two_bouts_excel: Path, tmp_path: Path
) -> None:
    sg_path = tmp_path / "state_groups.yaml"
    _state_groups_yaml(markov_two_bouts_excel, sg_path)
    fin_path = tmp_path / "finalists.yaml"
    fin_path.write_text("weight_classes: {}\n", encoding="utf-8")

    out_dir = tmp_path / "reports_empty"
    summary = build_aggregate_models(
        excel_path=markov_two_bouts_excel,
        state_groups_path=sg_path,
        output_dir=out_dir,
        finalists_path=fin_path,
        sheet="Общее",
    )
    assert summary.weight_classes_rendered == 0
    # index.html всё равно создаётся (с пустым списком категорий).
    assert (out_dir / "index.html").exists()
