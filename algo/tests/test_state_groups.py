"""Тесты загрузчика state_groups.yaml (TASK_SPEC_011, Фаза A).

Проверяем не путь «всё хорошо», а граничные случаи: неизвестные имена
состояний, ``pause`` с колонками, обе формы записи (плоский список и
``{columns: [...]}``), валидация против реального flatten листа.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hpc_algo.schema import EpisodeState, MarkovWarning
from hpc_algo.state_groups import (
    StateGroupsConfig,
    load_state_groups,
    validate_columns_against_sheet,
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "state_groups.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_state_groups_dict_form(tmp_path: Path) -> None:
    yaml_text = """
version: "1"
sheet: "Общее"
mode: single
priority:
  - technical_action
  - off_balance
  - grip
  - manoeuvring
  - pause
states:
  manoeuvring:
    columns:
      - "Стойка | ПС | Вперед"
  grip:
    columns:
      - "КФВ | Захваты | О"
  off_balance:
    columns:
      - "ВУП | ВУП-Р"
  technical_action:
    columns:
      - "Завершающие | Удержание"
"""
    cfg, warnings = load_state_groups(_write_yaml(tmp_path, yaml_text))

    assert isinstance(cfg, StateGroupsConfig)
    assert cfg.sheet == "Общее"
    assert cfg.mode == "single"
    assert cfg.priority[0] == EpisodeState.TECHNICAL_ACTION.value
    assert cfg.states["manoeuvring"] == ["Стойка | ПС | Вперед"]
    assert cfg.states["technical_action"] == ["Завершающие | Удержание"]
    assert warnings == []


def test_load_state_groups_flat_list_form(tmp_path: Path) -> None:
    """Плоский список (без `columns:`) — также поддерживается."""

    yaml_text = """
version: "1"
sheet: "Общее"
mode: single
states:
  manoeuvring:
    - "A"
    - "B"
  grip:
    - "C"
"""
    cfg, warnings = load_state_groups(_write_yaml(tmp_path, yaml_text))
    assert cfg.states["manoeuvring"] == ["A", "B"]
    assert cfg.states["grip"] == ["C"]
    assert warnings == []


def test_unknown_state_emits_warning(tmp_path: Path) -> None:
    yaml_text = """
states:
  manoeuvring: ["a"]
  unknown_state: ["x"]
"""
    cfg, warnings = load_state_groups(_write_yaml(tmp_path, yaml_text))
    codes = {w.code for w in warnings}
    assert "state_groups.unknown_state" in codes
    assert "unknown_state" in cfg.states


def test_pause_with_columns_emits_warning(tmp_path: Path) -> None:
    yaml_text = """
states:
  pause: ["foo"]
"""
    _, warnings = load_state_groups(_write_yaml(tmp_path, yaml_text))
    codes = {w.code for w in warnings}
    assert "state_groups.pause_has_columns" in codes


def test_invalid_mode_falls_back_to_single(tmp_path: Path) -> None:
    yaml_text = """
mode: triple
states:
  manoeuvring: []
"""
    cfg, warnings = load_state_groups(_write_yaml(tmp_path, yaml_text))
    assert cfg.mode == "single"
    codes = {w.code for w in warnings}
    assert "state_groups.invalid_mode" in codes


def test_unknown_priority_emits_warning(tmp_path: Path) -> None:
    yaml_text = """
priority:
  - technical_action
  - mystery_state
states: {}
"""
    cfg, warnings = load_state_groups(_write_yaml(tmp_path, yaml_text))
    codes = {w.code for w in warnings}
    assert "state_groups.unknown_priority" in codes
    assert "mystery_state" in cfg.priority


def test_validate_columns_against_sheet() -> None:
    cfg = StateGroupsConfig(
        states={
            "manoeuvring": ["KNOWN_COL", "MISSING_COL"],
            "grip": ["ANOTHER_KNOWN"],
        }
    )
    warnings = validate_columns_against_sheet(cfg, ["KNOWN_COL", "ANOTHER_KNOWN"])
    assert all(isinstance(w, MarkovWarning) for w in warnings)
    assert len(warnings) == 1
    assert warnings[0].code == "state_groups.column_not_found"
    assert warnings[0].context["column"] == "MISSING_COL"


def test_load_state_groups_invalid_root_raises(tmp_path: Path) -> None:
    """Если YAML — это не mapping, а, например, список — поднимаем ValueError."""

    p = tmp_path / "bad.yaml"
    p.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_state_groups(p)
