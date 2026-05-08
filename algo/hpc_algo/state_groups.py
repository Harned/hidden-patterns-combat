"""YAML-конфиг «группа состояния → flatten-колонки» для 5-state Marков-цепи.

Назначение модуля — изолировать чтение конфига от остальной логики и
вернуть структурированный результат без падений: любые расхождения с
ожидаемым форматом превращаются в :class:`MarkovWarning`, а не в
исключение, чтобы пайплайн мог продолжить с пустыми группами и честно
сообщить пользователю, что нужно поправить.

См. ``docs/agent_context/TASK_SPEC_011_INDIVIDUAL_MARKOV.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hpc_algo.schema import EpisodeState, MarkovMode, MarkovWarning

_DEFAULT_PRIORITY: tuple[str, ...] = (
    EpisodeState.TECHNICAL_ACTION.value,
    EpisodeState.OFF_BALANCE.value,
    EpisodeState.GRIP.value,
    EpisodeState.MANOEUVRING.value,
    EpisodeState.PAUSE.value,
)


class StateGroupsConfig(BaseModel):
    """Конфиг 5-state алфавита.

    ``states`` — словарь ``state_value → list[flatten_column_name]``.
    Состояние ``pause`` НЕ должно иметь колонок (определяется по отсутствию
    активности в остальных группах).
    """

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    sheet: str = "Общее"
    mode: MarkovMode = "single"
    priority: list[str] = Field(default_factory=lambda: list(_DEFAULT_PRIORITY))
    states: dict[str, list[str]] = Field(default_factory=dict)


def _coerce_columns(value: Any) -> list[str]:
    """Принять обе формы YAML: ``state: [col1, col2]`` и ``state: {columns: [...]}``."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict) and "columns" in value:
        cols = value.get("columns") or []
        if isinstance(cols, list):
            return [str(c) for c in cols]
    raise TypeError(value)


def load_state_groups(path: str | Path) -> tuple[StateGroupsConfig, list[MarkovWarning]]:
    """Прочитать YAML и вернуть нормализованный конфиг + warnings.

    Сигнатура осознанно «толерантная»: при структурных проблемах возвращаем
    конфиг с пустыми группами и warnings, чтобы вызвавший слой мог решать,
    падать ли. Падаем только если файл нечитаем как YAML вообще.
    """

    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)

    warnings: list[MarkovWarning] = []
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"State groups config must be a YAML mapping, got: {type(raw).__name__}"
        )

    states_raw = raw.get("states", {}) or {}
    states_normalized: dict[str, list[str]] = {}
    if isinstance(states_raw, dict):
        for name, value in states_raw.items():
            try:
                states_normalized[str(name)] = _coerce_columns(value)
            except TypeError:
                warnings.append(
                    MarkovWarning(
                        code="state_groups.invalid_entry",
                        message=(
                            f"State '{name}' has invalid columns spec; "
                            "expected list[str] or dict with 'columns'."
                        ),
                        context={"state": str(name)},
                    )
                )
                states_normalized[str(name)] = []
    else:
        warnings.append(
            MarkovWarning(
                code="state_groups.invalid_states_root",
                message="`states` must be a YAML mapping; got a different type.",
            )
        )

    mode_value = raw.get("mode", "single")
    if mode_value not in ("single", "multi"):
        warnings.append(
            MarkovWarning(
                code="state_groups.invalid_mode",
                message=f"Unknown mode '{mode_value}', falling back to 'single'.",
                context={"mode": str(mode_value)},
            )
        )
        mode_value = "single"

    priority_raw = raw.get("priority")
    if priority_raw is None:
        priority = list(_DEFAULT_PRIORITY)
    elif isinstance(priority_raw, list):
        priority = [str(p) for p in priority_raw]
    else:
        warnings.append(
            MarkovWarning(
                code="state_groups.invalid_priority_type",
                message="`priority` must be a list of state names.",
            )
        )
        priority = list(_DEFAULT_PRIORITY)

    cfg = StateGroupsConfig(
        version=str(raw.get("version", "1")),
        sheet=str(raw.get("sheet", "Общее")),
        mode=mode_value,
        priority=priority,
        states=states_normalized,
    )

    valid_state_values = {s.value for s in EpisodeState}
    for name in cfg.states:
        if name not in valid_state_values:
            warnings.append(
                MarkovWarning(
                    code="state_groups.unknown_state",
                    message=f"Unknown state name in YAML: {name}",
                    context={"state": name, "allowed": sorted(valid_state_values)},
                )
            )

    if cfg.states.get(EpisodeState.PAUSE.value):
        warnings.append(
            MarkovWarning(
                code="state_groups.pause_has_columns",
                message=(
                    "'pause' must not have columns; pause is determined by"
                    " absence of activity in other groups."
                ),
            )
        )

    for name in cfg.priority:
        if name not in valid_state_values:
            warnings.append(
                MarkovWarning(
                    code="state_groups.unknown_priority",
                    message=f"Unknown state in priority: {name}",
                    context={"state": name, "allowed": sorted(valid_state_values)},
                )
            )

    return cfg, warnings


def validate_columns_against_sheet(
    cfg: StateGroupsConfig,
    flatten_columns: list[str],
) -> list[MarkovWarning]:
    """Сверить колонки YAML с реальными flatten-именами листа.

    Каждая отсутствующая колонка превращается в warning
    ``state_groups.column_not_found``; сама ошибка не фатальна — модель
    просто построится с меньшим числом источников активности.
    """

    warnings: list[MarkovWarning] = []
    available = set(flatten_columns)
    for state, cols in cfg.states.items():
        for col in cols:
            if col not in available:
                warnings.append(
                    MarkovWarning(
                        code="state_groups.column_not_found",
                        message=(
                            f"Column '{col}' (state={state}) not found in sheet"
                            " after flatten."
                        ),
                        context={"state": state, "column": col},
                    )
                )
    return warnings
