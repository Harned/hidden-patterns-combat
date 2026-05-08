"""Тесты модуля :mod:`hpc_algo.episode_metrics`.

Покрытие (минимум, без classify_style — он отложен в TASK_SPEC_013):

* пустой вход → все поля по умолчанию;
* duration / pause stats считают `mean / median / std / total`;
* action_density = sum(features>0) / total_time, делит правильно при
  единственном валидном эпизоде;
* non_technical_share игнорирует `technical_action`, считая всех
  остальных (в т.ч. ``pause``);
* activity_evenness в `[0, 1]`, ровно 1.0 на равномерном распределении;
* фильтр ``athlete`` корректно сужает обе коллекции.
"""

from __future__ import annotations

import math

from hpc_algo.episode_metrics import compute_episode_metrics
from hpc_algo.episode_split import RawEpisode
from hpc_algo.schema import EpisodeRecord, EpisodeState


def _raw(
    *,
    athlete: str,
    bout_id: str,
    idx: int,
    ep_time: float | None,
    pause_time: float | None,
    features: dict[str, float] | None = None,
    row_index: int = 0,
) -> RawEpisode:
    return RawEpisode(
        row_index=row_index,
        bout_id=bout_id,
        episode_idx_in_bout=idx,
        athlete=athlete,
        episode_num_raw=float(idx),
        episode_time=ep_time,
        pause_time=pause_time,
        score=None,
        feature_values=features or {},
    )


def _rec(*, athlete: str, bout_id: str, idx: int, state: EpisodeState) -> EpisodeRecord:
    return EpisodeRecord(
        athlete=athlete,
        bout_id=bout_id,
        episode_idx=idx,
        state=state,
    )


def test_empty_inputs_produce_neutral_metrics() -> None:
    m = compute_episode_metrics([], [])
    assert m.episode_count == 0
    assert m.bout_count == 0
    assert m.duration_stats.count == 0
    assert m.duration_stats.mean is None
    assert m.action_density is None
    assert m.non_technical_share is None
    assert m.activity_evenness is None


def test_duration_and_density_basic() -> None:
    raw = [
        _raw(athlete="A", bout_id="b1", idx=1, ep_time=10.0, pause_time=2.0,
             features={"col_x": 1.0, "col_y": 0.0}),
        _raw(athlete="A", bout_id="b1", idx=2, ep_time=20.0, pause_time=3.0,
             features={"col_x": 2.0, "col_y": 1.0}),
    ]
    records = [
        _rec(athlete="A", bout_id="b1", idx=1, state=EpisodeState.MANOEUVRING),
        _rec(athlete="A", bout_id="b1", idx=2, state=EpisodeState.TECHNICAL_ACTION),
    ]

    m = compute_episode_metrics(raw, records, athlete="A")

    assert m.episode_count == 2
    assert m.bout_count == 1

    d = m.duration_stats
    assert d.count == 2
    assert d.total == 30.0
    assert math.isclose(d.mean or 0.0, 15.0)

    p = m.pause_stats
    assert p.count == 2
    assert p.total == 5.0

    # Действий: 1 + 0 + 2 + 1 = 4; время: 30; density = 4/30
    assert m.action_density is not None
    assert math.isclose(m.action_density, 4.0 / 30.0, rel_tol=1e-6)


def test_non_technical_share_includes_pause() -> None:
    records = [
        _rec(athlete="A", bout_id="b1", idx=1, state=EpisodeState.MANOEUVRING),
        _rec(athlete="A", bout_id="b1", idx=2, state=EpisodeState.PAUSE),
        _rec(athlete="A", bout_id="b1", idx=3, state=EpisodeState.TECHNICAL_ACTION),
    ]
    m = compute_episode_metrics([], records)
    # 2 не-ЗАП из 3 → 2/3
    assert m.non_technical_share is not None
    assert math.isclose(m.non_technical_share, 2 / 3, rel_tol=1e-6)


def test_evenness_extremes() -> None:
    # Равномерное распределение по 5 состояниям → 1.0
    records_uniform = [
        _rec(athlete="A", bout_id="b1", idx=i + 1, state=s)
        for i, s in enumerate(
            [
                EpisodeState.MANOEUVRING,
                EpisodeState.GRIP,
                EpisodeState.OFF_BALANCE,
                EpisodeState.TECHNICAL_ACTION,
                EpisodeState.PAUSE,
            ]
        )
    ]
    m_uniform = compute_episode_metrics([], records_uniform)
    assert m_uniform.activity_evenness is not None
    assert math.isclose(m_uniform.activity_evenness, 1.0, rel_tol=1e-6)

    # Всё в одном состоянии → 0.0
    records_single = [
        _rec(athlete="A", bout_id="b1", idx=i + 1, state=EpisodeState.PAUSE)
        for i in range(5)
    ]
    m_single = compute_episode_metrics([], records_single)
    assert m_single.activity_evenness == 0.0


def test_athlete_filter_isolates_subject() -> None:
    raw = [
        _raw(athlete="A", bout_id="bA", idx=1, ep_time=10.0, pause_time=1.0,
             features={"x": 1.0}),
        _raw(athlete="B", bout_id="bB", idx=1, ep_time=20.0, pause_time=2.0,
             features={"x": 1.0}),
    ]
    records = [
        _rec(athlete="A", bout_id="bA", idx=1, state=EpisodeState.MANOEUVRING),
        _rec(athlete="B", bout_id="bB", idx=1, state=EpisodeState.TECHNICAL_ACTION),
    ]
    m_a = compute_episode_metrics(raw, records, athlete="A")
    assert m_a.episode_count == 1
    assert m_a.bout_count == 1
    assert m_a.duration_stats.total == 10.0


def test_invalid_durations_filtered_out() -> None:
    raw = [
        _raw(athlete="A", bout_id="b", idx=1, ep_time=float("nan"), pause_time=None),
        _raw(athlete="A", bout_id="b", idx=2, ep_time=-5.0, pause_time=2.0),
        _raw(athlete="A", bout_id="b", idx=3, ep_time=10.0, pause_time=None),
    ]
    records = [
        _rec(athlete="A", bout_id="b", idx=i + 1, state=EpisodeState.MANOEUVRING)
        for i in range(3)
    ]
    m = compute_episode_metrics(raw, records, athlete="A")
    # Только одно валидное значение длительности эпизода (10.0).
    assert m.duration_stats.count == 1
    assert m.duration_stats.total == 10.0
    # Только одна валидная пауза (2.0).
    assert m.pause_stats.count == 1
    assert m.pause_stats.total == 2.0
