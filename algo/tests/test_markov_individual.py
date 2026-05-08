"""Тесты индивидуальной 5-state Marков-цепи (TASK_SPEC_011, Фаза A).

Покрываем:

* приоритет ``technical_action > off_balance > grip > manoeuvring > pause``
  в режиме ``single``;
* раскрытие эпизода в `multi`-режиме;
* инвариант ``sum(row) = 1 ± 1e-6`` для матрицы переходов;
* отсутствие cross-bout переходов;
* строка матрицы для ненаблюдавшегося «откуда» — равномерная + warning;
* fallback time-averaging при разреженных данных + warning ``mc.non_ergodic``;
* end-to-end сборка от Excel до :class:`MarkovIndividualResult` на
  фикстуре ``markov_two_bouts_excel``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hpc_algo.episode_split import (
    BaseColumns,
    detect_base_columns,
    read_episodes_sheet,
    split_into_bouts_and_episodes,
)
from hpc_algo.markov_individual import (
    assign_state_multi,
    assign_state_single,
    build_episode_sequence,
    fit_individual_markov,
)
from hpc_algo.schema import EpisodeRecord, EpisodeState
from hpc_algo.state_groups import StateGroupsConfig


def _make_cfg(mode: str = "single") -> StateGroupsConfig:
    return StateGroupsConfig(
        version="1",
        sheet="X",
        mode=mode,
        priority=[s.value for s in (
            EpisodeState.TECHNICAL_ACTION,
            EpisodeState.OFF_BALANCE,
            EpisodeState.GRIP,
            EpisodeState.MANOEUVRING,
            EpisodeState.PAUSE,
        )],
        states={
            "manoeuvring": ["m1"],
            "grip": ["g1"],
            "off_balance": ["o1"],
            "technical_action": ["t1"],
        },
    )


# --- assign_state ---


def test_single_priority_grip_beats_manoeuvring() -> None:
    cfg = _make_cfg("single")
    state = assign_state_single({"m1": 1, "g1": 1, "o1": 0, "t1": 0}, cfg)
    assert state == EpisodeState.GRIP


def test_single_priority_technical_beats_all() -> None:
    cfg = _make_cfg("single")
    state = assign_state_single({"m1": 1, "g1": 1, "o1": 1, "t1": 1}, cfg)
    assert state == EpisodeState.TECHNICAL_ACTION


def test_single_priority_pause_when_inactive() -> None:
    cfg = _make_cfg("single")
    state = assign_state_single({"m1": 0, "g1": 0, "o1": 0, "t1": 0}, cfg)
    assert state == EpisodeState.PAUSE


def test_multi_unfolds_in_domain_order() -> None:
    cfg = _make_cfg("multi")
    seq = assign_state_multi({"m1": 1, "g1": 0, "o1": 0, "t1": 1}, cfg)
    assert seq == [EpisodeState.MANOEUVRING, EpisodeState.TECHNICAL_ACTION]


def test_multi_pause_when_inactive() -> None:
    cfg = _make_cfg("multi")
    seq = assign_state_multi({"m1": 0, "g1": 0, "o1": 0, "t1": 0}, cfg)
    assert seq == [EpisodeState.PAUSE]


# --- fit_individual_markov ---


def _records_from_states(
    athlete: str,
    bouts: list[list[EpisodeState]],
) -> list[EpisodeRecord]:
    out: list[EpisodeRecord] = []
    for bout_idx, seq in enumerate(bouts, start=1):
        bout_id = f"bout_{bout_idx}"
        for ep_idx, st in enumerate(seq, start=1):
            out.append(
                EpisodeRecord(
                    athlete=athlete,
                    bout_id=bout_id,
                    episode_idx=ep_idx,
                    state=st,
                )
            )
    return out


def test_row_sum_invariant_holds() -> None:
    records = _records_from_states(
        "X",
        [
            [EpisodeState.MANOEUVRING, EpisodeState.GRIP, EpisodeState.TECHNICAL_ACTION],
            [EpisodeState.MANOEUVRING, EpisodeState.OFF_BALANCE, EpisodeState.TECHNICAL_ACTION],
        ],
    )
    result = fit_individual_markov("X", records)
    A = np.asarray(result.transition_matrix)
    row_sums = A.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_no_cross_bout_transitions() -> None:
    """Граница bout'а не порождает переход."""

    records = _records_from_states(
        "X",
        [
            [EpisodeState.MANOEUVRING, EpisodeState.GRIP],
            [EpisodeState.TECHNICAL_ACTION, EpisodeState.PAUSE],
        ],
    )
    result = fit_individual_markov("X", records)
    counts = np.asarray(result.transition_counts)
    states = list(EpisodeState)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Внутри bout 1: manoeuvring → grip
    assert counts[state_to_idx[EpisodeState.MANOEUVRING],
                  state_to_idx[EpisodeState.GRIP]] == 1
    # Внутри bout 2: technical_action → pause
    assert counts[state_to_idx[EpisodeState.TECHNICAL_ACTION],
                  state_to_idx[EpisodeState.PAUSE]] == 1
    # Никаких переходов grip → technical_action (это была бы cross-bout)
    assert counts[state_to_idx[EpisodeState.GRIP],
                  state_to_idx[EpisodeState.TECHNICAL_ACTION]] == 0


def test_unobserved_row_filled_uniformly_and_warned() -> None:
    """Состояние, которое не появилось как «откуда», даёт равномерную строку + warning."""

    records = _records_from_states(
        "X",
        [
            # Только manoeuvring → grip → manoeuvring; остальные «откуда» не наблюдались.
            [EpisodeState.MANOEUVRING, EpisodeState.GRIP, EpisodeState.MANOEUVRING],
        ],
    )
    result = fit_individual_markov("X", records)
    A = np.asarray(result.transition_matrix)
    states = list(EpisodeState)
    state_to_idx = {s: i for i, s in enumerate(states)}

    row_pause = A[state_to_idx[EpisodeState.PAUSE]]
    assert np.allclose(row_pause, np.full(len(states), 1.0 / len(states)))

    codes = {w.code for w in result.warnings}
    assert "mc.unobserved_row" in codes


def test_non_ergodic_fallback_when_data_is_sparse() -> None:
    """≥ 2 состояний без посещений → time-averaging + warning."""

    records = _records_from_states(
        "X",
        [
            [EpisodeState.MANOEUVRING, EpisodeState.MANOEUVRING, EpisodeState.MANOEUVRING],
        ],
    )
    result = fit_individual_markov("X", records)
    codes = {w.code for w in result.warnings}
    assert "mc.non_ergodic" in codes
    pi = np.asarray(result.stationary)
    # Только manoeuvring наблюдалось → time-averaging должен дать 1.0 на manoeuvring.
    states = list(EpisodeState)
    state_to_idx = {s: i for i, s in enumerate(states)}
    assert pi[state_to_idx[EpisodeState.MANOEUVRING]] == 1.0


def test_fit_filters_by_athlete() -> None:
    records = _records_from_states("A", [[EpisodeState.GRIP, EpisodeState.GRIP]])
    records += _records_from_states("B", [[EpisodeState.TECHNICAL_ACTION]])
    result_a = fit_individual_markov("A", records)
    result_b = fit_individual_markov("B", records)
    assert result_a.episode_count == 2
    assert result_b.episode_count == 1


# --- end-to-end ---


def test_end_to_end_on_markov_two_bouts(markov_two_bouts_excel: Path) -> None:
    """Сквозной прогон: read → split → assign → fit для одного спортсмена."""

    df = read_episodes_sheet(markov_two_bouts_excel, sheet="Общее")
    base = detect_base_columns(df.columns)
    assert isinstance(base, BaseColumns)

    feature_cols = [c for c in df.columns if c not in {
        base.athlete, base.episode_num, base.episode_time, base.pause_time, base.score
    }]
    raw_eps, _ = split_into_bouts_and_episodes(df, base, feature_cols)

    cfg = StateGroupsConfig(
        version="1",
        sheet="Общее",
        mode="single",
        priority=[s.value for s in (
            EpisodeState.TECHNICAL_ACTION,
            EpisodeState.OFF_BALANCE,
            EpisodeState.GRIP,
            EpisodeState.MANOEUVRING,
            EpisodeState.PAUSE,
        )],
        states={
            "manoeuvring": [
                c for c in df.columns if "Стойка" in c
            ],
            "grip": [
                c for c in df.columns if "КФВ" in c
            ],
            "off_balance": [
                c for c in df.columns if "ВУП" in c
            ],
            "technical_action": [
                c for c in df.columns if "Завершающие" in c
            ],
        },
    )
    records = build_episode_sequence(raw_eps, cfg)

    result = fit_individual_markov("Иванов", records, mode="single")
    assert result.bout_count == 2
    assert result.episode_count == 5  # 3 + 2

    A = np.asarray(result.transition_matrix)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)
