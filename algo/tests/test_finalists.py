"""Тесты YAML-загрузчика и эвристического детекта призёров (TS_012)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hpc_algo.finalists import (
    detect_meta_columns,
    finalists_from_episodes,
    load_finalists_yaml,
)


def test_load_finalists_yaml_basic(tmp_path: Path) -> None:
    p = tmp_path / "finalists.yaml"
    p.write_text(
        """
version: 1
weight_classes:
  "48":
    1: "Иванов И. И."
    2: "Петров П."
    3: "Сидоров С."
  "52":
    1: "А."
    2: "Б."
""",
        encoding="utf-8",
    )
    entries, warnings = load_finalists_yaml(p)
    assert warnings == []
    by_class: dict[str, dict[int, str]] = {}
    for e in entries:
        by_class.setdefault(e.weight_class, {})[e.place] = e.athlete
    assert by_class["48"] == {
        1: "Иванов И. И.",
        2: "Петров П.",
        3: "Сидоров С.",
    }
    assert by_class["52"] == {1: "А.", 2: "Б."}


def test_load_finalists_yaml_flat_form(tmp_path: Path) -> None:
    p = tmp_path / "finalists.yaml"
    p.write_text(
        """
"48":
  1: "Иванов"
  2: "Петров"
""",
        encoding="utf-8",
    )
    entries, warnings = load_finalists_yaml(p)
    assert warnings == []
    assert len(entries) == 2


def test_load_finalists_yaml_warns_on_invalid_place(tmp_path: Path) -> None:
    p = tmp_path / "finalists.yaml"
    p.write_text(
        """
weight_classes:
  "48":
    "first": "Иванов"
    2: ""
""",
        encoding="utf-8",
    )
    entries, warnings = load_finalists_yaml(p)
    codes = {w.code for w in warnings}
    assert "finalists.invalid_place" in codes
    assert "finalists.empty_athlete" in codes
    assert entries == []


def test_load_finalists_yaml_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "finalists.yaml"
    p.write_text("", encoding="utf-8")
    entries, warnings = load_finalists_yaml(p)
    assert entries == []
    assert warnings == []


def test_load_finalists_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "finalists.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_finalists_yaml(p)


def test_detect_meta_columns_finds_both() -> None:
    cols = [
        "ФИО борца",
        "Категория | Весовая категория, кг",
        "Результат | Место",
        "Стойка | ...",
    ]
    weight, place = detect_meta_columns(cols)
    assert weight == "Категория | Весовая категория, кг"
    assert place == "Результат | Место"


def test_detect_meta_columns_returns_none_when_absent() -> None:
    weight, place = detect_meta_columns(["ФИО борца", "Стойка | ПС"])
    assert weight is None
    assert place is None


def test_finalists_from_episodes_warns_when_columns_missing() -> None:
    entries, warnings = finalists_from_episodes(
        raw_episodes=[],
        df_columns=["ФИО борца", "Стойка | ПС"],
        df_rows=[],
    )
    codes = {w.code for w in warnings}
    assert "finalists.columns_missing" in codes
    assert entries == []
