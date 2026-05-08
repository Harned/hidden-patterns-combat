"""Поэпизодные метрики управления поединком (частичная реализация TASK_SPEC_013).

Сейчас покрыты: ``episode_count``, ``bout_count``, ``duration_stats``
(по эпизоду), ``pause_stats``, ``action_density``, ``non_technical_share``,
``activity_evenness``.

`classify_style` (`endurance | speed_power | burnout`) — отдельный шаг
по ``TASK_SPEC_013``: требует YAML-порогов и описательного словаря,
поэтому намеренно отложен. Поля :class:`hpc_algo.schema.EpisodeMetrics`
расширяются, но текущие НЕ переименовываются и НЕ перетипизируются.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from hpc_algo.episode_split import RawEpisode
from hpc_algo.schema import (
    DurationStats,
    EpisodeMetrics,
    EpisodeRecord,
    EpisodeState,
)


def _duration_stats(values: list[float]) -> DurationStats:
    valid = [v for v in values if v is not None and math.isfinite(v) and v >= 0]
    if not valid:
        return DurationStats()
    if len(valid) == 1:
        only = float(valid[0])
        return DurationStats(
            count=1,
            mean=only,
            median=only,
            std=0.0,
            min=only,
            max=only,
            total=only,
        )
    return DurationStats(
        count=len(valid),
        mean=float(statistics.fmean(valid)),
        median=float(statistics.median(valid)),
        std=float(statistics.pstdev(valid)),
        min=float(min(valid)),
        max=float(max(valid)),
        total=float(sum(valid)),
    )


def _record_first_state(rec: EpisodeRecord) -> EpisodeState | None:
    if isinstance(rec.state, list):
        return rec.state[0] if rec.state else None
    return rec.state


def _evenness(visit_counts: dict[str, int]) -> float | None:
    """Нормализованная энтропия Шеннона для распределения visit_counts.

    Возвращает None, если сумма счётчиков 0 или ровно одно состояние
    наблюдалось (энтропия = 0, но интерпретировать как «равномерность 0»
    осмысленно — поэтому в этом случае возвращаем 0.0, а None оставляем
    только для пустых данных).
    """

    total = sum(visit_counts.values())
    if total <= 0:
        return None
    n_nonzero = sum(1 for c in visit_counts.values() if c > 0)
    if n_nonzero <= 1:
        return 0.0
    h = 0.0
    for c in visit_counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    n_states = len(visit_counts)
    if n_states <= 1:
        return 0.0
    return float(h / math.log(n_states))


def compute_episode_metrics(
    raw_episodes: Iterable[RawEpisode],
    records: Iterable[EpisodeRecord],
    *,
    athlete: str | None = None,
) -> EpisodeMetrics:
    """Посчитать описательные метрики по выборке эпизодов одного спортсмена.

    Параметры:

    * ``raw_episodes`` — нормализованные строки листа (для длительностей,
      пауз, action_density по сырым ``feature_values``).
    * ``records`` — построенная последовательность ``EpisodeRecord``
      (для ``non_technical_share`` по реально присвоенным состояниям).
    * ``athlete`` — если задан, оба источника фильтруются по точному
      совпадению ``athlete``; иначе считаем по всем переданным записям.

    Замечание о действиях/секунду: сумма ``feature_values`` по эпизоду
    интерпретируется как количество отдельных действий (значение ``2``
    означает «дважды» — поэтому даёт +2). Время — суммарное по эпизодам,
    у которых ``episode_time`` валиден.
    """

    raw_list = [e for e in raw_episodes if athlete is None or e.athlete == athlete]
    rec_list = [r for r in records if athlete is None or r.athlete == athlete]

    episode_durations = [e.episode_time for e in raw_list if e.episode_time is not None]
    pause_durations = [e.pause_time for e in raw_list if e.pause_time is not None]

    duration_stats = _duration_stats(episode_durations)
    pause_stats = _duration_stats(pause_durations)

    action_density: float | None = None
    total_time = duration_stats.total or 0.0
    if total_time > 0:
        total_actions = 0.0
        for e in raw_list:
            for v in e.feature_values.values():
                if v > 0:
                    total_actions += float(v)
        action_density = float(total_actions / total_time)

    non_technical_share: float | None = None
    if rec_list:
        first_states = [_record_first_state(r) for r in rec_list]
        non_technical = sum(
            1 for s in first_states if s is not None and s != EpisodeState.TECHNICAL_ACTION
        )
        non_technical_share = float(non_technical / len(rec_list))

    visit_counts: dict[str, int] = {s.value: 0 for s in EpisodeState}
    for r in rec_list:
        states = r.state if isinstance(r.state, list) else [r.state]
        for s in states:
            visit_counts[s.value] += 1
    activity_evenness = _evenness(visit_counts)

    bout_count = len({e.bout_id for e in raw_list})

    return EpisodeMetrics(
        athlete=athlete,
        episode_count=len(rec_list) if rec_list else len(raw_list),
        bout_count=bout_count,
        duration_stats=duration_stats,
        pause_stats=pause_stats,
        action_density=action_density,
        non_technical_share=non_technical_share,
        activity_evenness=activity_evenness,
    )
