"""Регрессии для виртуального bout, восстановленного по сбросам нумерации эпизодов.

Реальные файлы вида ``docs/Оценка СД содержание.xlsx`` обычно не содержат
явной колонки «Схватка» — нумерация эпизодов перезапускается с 1 в
каждой новой схватке. Без бойтовой разметки HMM-серия = все эпизоды
одного борца, что искажает структуру переходов и иногда не проходит
``min_sequence_median``-guard. Phase 2 плана автоматически вставляет
виртуальный bout в mapping (с явным INFO-warning), при условии, что
медиана длины серии после применения ≥ 2.

Тесты проверяют:

* :func:`hpc_algo.mapping.virtual_bout_series` корректно увеличивает
  счётчик при сбросе номера эпизода;
* при отсутствии роли ``bout`` ``analyze_source`` подменяет mapping
  виртуальной колонкой и эмитит warning ``hmm.bout_inferred_from_episode_resets``;
* семантика серий после применения соответствует структуре фикстуры
  (4 борца × 2 виртуальных схватки = 8 серий длины 3).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.mapping import virtual_bout_series
from hpc_algo.schema import HiddenGroup


def test_virtual_bout_series_increments_on_reset() -> None:
    df = pd.DataFrame(
        {
            "athlete": ["A", "A", "A", "A", "B", "B", "B"],
            "episode": [1, 2, 3, 1, 1, 2, 1],
        }
    )
    series = virtual_bout_series(df, "athlete", "episode")
    assert series is not None
    # A: 1,1,1,2 (сброс 3 → 1 запускает новый bout)
    # B: 1,1,2 (сброс 2 → 1 запускает новый bout)
    assert list(series) == [1, 1, 1, 2, 1, 1, 2]


def test_virtual_bout_series_returns_none_when_episode_not_numeric() -> None:
    df = pd.DataFrame(
        {
            "athlete": ["A", "A", "B"],
            "episode": ["x", "y", "z"],
        }
    )
    assert virtual_bout_series(df, "athlete", "episode") is None


def test_analyze_emits_inferred_bout_warning(bout_resets_excel: Path) -> None:
    cfg = preflight_mapping(bout_resets_excel)
    sheet_name = next(iter(cfg.sheets))
    sm = cfg.sheets[sheet_name]
    # Подтверждаем preflight-ассампшн: bout роли нет.
    assert HiddenGroup.BOUT not in sm.roles or not sm.roles.get(HiddenGroup.BOUT)
    assert HiddenGroup.ATHLETE in sm.roles
    assert HiddenGroup.EPISODE in sm.roles

    result = analyze_source(bout_resets_excel, AnalyzeConfig(column_mapping=cfg))

    codes = {w.code for w in result.warnings}
    assert "hmm.bout_inferred_from_episode_resets" in codes, (
        f"Ожидаем INFO о виртуальном bout, получили: {codes}"
    )

    # Применённый mapping содержит синтетическую BOUT-колонку.
    assert result.applied_mapping is not None
    applied_sm = result.applied_mapping.sheets[sheet_name]
    assert applied_sm.roles.get(HiddenGroup.BOUT), applied_sm.roles
    bout_col = applied_sm.roles[HiddenGroup.BOUT][0]
    assert bout_col.startswith("_virtual_bout"), bout_col

    # При успешном применении HMM-серий ровно столько, сколько
    # виртуальных боёв (4 борца × 2 схватки = 8), и медиана длины ≥ 2.
    info = next(
        w.context for w in result.warnings
        if w.code == "hmm.bout_inferred_from_episode_resets"
    )
    assert info["n_sequences"] == 8
    assert info["median_sequence_length"] >= 2
    assert info["n_virtual_bouts"] == 2


def test_analyze_keeps_user_bout_mapping(bout_resets_excel: Path) -> None:
    """Если пользователь явно разметил bout, виртуальный bout не подменяется."""

    cfg = preflight_mapping(bout_resets_excel)
    sheet_name = next(iter(cfg.sheets))
    sm = cfg.sheets[sheet_name]
    # Имитируем явный bout: возьмём колонку «Время эпизода» как заглушку
    # (в тесте важна только установка роли, не семантика значений).
    time_cols = sm.roles.get(HiddenGroup.TIME) or []
    if not time_cols:
        return  # фикстура без времени; пропускаем
    new_roles = {role: list(cols) for role, cols in sm.roles.items()}
    new_roles[HiddenGroup.BOUT] = [time_cols[0]]
    cfg.sheets[sheet_name] = sm.model_copy(update={"roles": new_roles})

    result = analyze_source(bout_resets_excel, AnalyzeConfig(column_mapping=cfg))
    codes = {w.code for w in result.warnings}
    assert "hmm.bout_inferred_from_episode_resets" not in codes
