"""Поэпизодные метрики и описательная классификация стиля (TASK_SPEC_013).

Содержит две публичные функции:

* :func:`compute_episode_metrics` — пересчитывает базовые описательные
  метрики по эпизодному потоку спортсмена.
* :func:`classify_style` — пороговый классификатор стиля управления
  эпизодом (`endurance | speed_power | burnout | unclassified`).
  Пороги — внешний YAML (см. :func:`load_style_thresholds`).

Метрики — описательное расширение, не диагноз и не замена матрицы
переходов. Никаких ссылок на HMM.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from hpc_algo.episode_split import RawEpisode
from hpc_algo.schema import (
    DurationStats,
    EpisodeMetrics,
    EpisodeRecord,
    EpisodeState,
    MarkovWarning,
    StyleLabel,
)


def _percentile(values: list[float], q: float) -> float:
    """Линейная интерполяция между порядковыми статистиками (q ∈ [0, 100])."""

    if not values:
        raise ValueError("empty values")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _duration_stats(values: list[float]) -> DurationStats:
    valid = [float(v) for v in values if v is not None and math.isfinite(v) and v >= 0]
    if not valid:
        return DurationStats()
    if len(valid) == 1:
        only = valid[0]
        return DurationStats(
            count=1,
            mean=only,
            median=only,
            std=0.0,
            p25=only,
            p75=only,
            min=only,
            max=only,
            total=only,
        )
    return DurationStats(
        count=len(valid),
        mean=float(statistics.fmean(valid)),
        median=float(statistics.median(valid)),
        std=float(statistics.pstdev(valid)),
        p25=_percentile(valid, 25.0),
        p75=_percentile(valid, 75.0),
        min=float(min(valid)),
        max=float(max(valid)),
        total=float(sum(valid)),
    )


def _record_first_state(rec: EpisodeRecord) -> EpisodeState | None:
    if isinstance(rec.state, list):
        return rec.state[0] if rec.state else None
    return rec.state


def _per_episode_actions(raw: RawEpisode) -> float:
    """Количество активаций в одном эпизоде (значения ``1`` и ``2``).

    ``>2`` уже отрезаны на уровне ``episode_split._normalize_feature_value``,
    поэтому сюда долетают только ``{0, 1, 2}``.
    """

    return float(sum(v for v in raw.feature_values.values() if v > 0))


def _evenness_of_distribution(values: list[float]) -> float | None:
    """Нормализованная энтропия распределения по эпизодам.

    На вход — массив неотрицательных действий-в-эпизоде. Сначала нормируем
    в распределение вероятностей, затем считаем H = −Σ p log p и делим на
    ``log(N_episodes)``. Возвращаем ``None``, если эпизодов меньше двух
    или сумма нулевая (нет действий).
    """

    n = len(values)
    if n < 2:
        return None
    total = sum(values)
    if total <= 0:
        return None
    h = 0.0
    for v in values:
        if v <= 0:
            continue
        p = v / total
        h -= p * math.log(p)
    return float(h / math.log(n))


def compute_episode_metrics(
    raw_episodes: Iterable[RawEpisode],
    records: Iterable[EpisodeRecord],
    *,
    athlete: str | None = None,
) -> EpisodeMetrics:
    """Посчитать описательные метрики по выборке эпизодов одного спортсмена.

    Поведение:

    * ``raw_episodes`` дают активность и длительности; ``records`` —
      присвоенные состояния. Если задан ``athlete``, оба источника
      фильтруются по точному совпадению ``athlete``.
    * ``action_density`` = ``Σ activations / N_episodes`` (TS_013).
      Если эпизодов нет — ``None``.
    * ``action_rate_per_second`` = ``Σ activations / Σ episode_time`` —
      описательная производная для отображения, не вход классификатора.
    * ``action_density_first_half`` / ``_second_half`` рассчитываются по
      эпизодному потоку, отсортированному по ``(bout_id, episode_idx_in_bout,
      row_index)``; если эпизодов меньше двух, обе половины ``None``.
    * ``activity_evenness`` — нормализованная энтропия per-episode
      ``activations``.
    * ``style`` оставляется ``None``; классификатор вызывает оркестратор
      отдельно через :func:`classify_style`.
    """

    raw_list = [e for e in raw_episodes if athlete is None or e.athlete == athlete]
    rec_list = [r for r in records if athlete is None or r.athlete == athlete]

    raw_list_sorted = sorted(
        raw_list,
        key=lambda e: (e.bout_id, e.episode_idx_in_bout, e.row_index),
    )

    episode_durations = [e.episode_time for e in raw_list if e.episode_time is not None]
    pause_durations = [e.pause_time for e in raw_list if e.pause_time is not None]
    duration_stats = _duration_stats(episode_durations)
    pause_stats = _duration_stats(pause_durations)

    per_ep_actions = [_per_episode_actions(e) for e in raw_list_sorted]

    n_episodes = len(per_ep_actions)
    total_actions = float(sum(per_ep_actions))

    action_density: float | None = None
    if n_episodes > 0:
        action_density = float(total_actions / n_episodes)

    action_rate_per_second: float | None = None
    total_time = duration_stats.total or 0.0
    if total_time > 0:
        action_rate_per_second = float(total_actions / total_time)

    first_half: float | None = None
    second_half: float | None = None
    if n_episodes >= 2:
        mid = n_episodes // 2
        first = per_ep_actions[:mid]
        second = per_ep_actions[mid:]
        if first:
            first_half = float(sum(first) / len(first))
        if second:
            second_half = float(sum(second) / len(second))

    activity_evenness = _evenness_of_distribution(per_ep_actions)

    non_technical_share: float | None = None
    if rec_list:
        first_states = [_record_first_state(r) for r in rec_list]
        non_technical = sum(
            1
            for s in first_states
            if s is not None and s != EpisodeState.TECHNICAL_ACTION
        )
        non_technical_share = float(non_technical / len(rec_list))

    bout_count = len({e.bout_id for e in raw_list_sorted})

    return EpisodeMetrics(
        athlete=athlete,
        episode_count=len(rec_list) if rec_list else n_episodes,
        bout_count=bout_count,
        duration_stats=duration_stats,
        pause_stats=pause_stats,
        action_density=action_density,
        action_rate_per_second=action_rate_per_second,
        action_density_first_half=first_half,
        action_density_second_half=second_half,
        non_technical_share=non_technical_share,
        activity_evenness=activity_evenness,
        style=None,
    )


# ---------------------------------------------------------------------------
# YAML-пороги стиля и классификатор
# ---------------------------------------------------------------------------

_KNOWN_STYLE_KEYS = {
    StyleLabel.ENDURANCE.value,
    StyleLabel.SPEED_POWER.value,
    StyleLabel.BURNOUT.value,
}

# Все поддерживаемые имена порогов. Любой другой ключ в YAML — warning,
# но не падение, чтобы конфиг не блокировал прогон отчётов.
_KNOWN_THRESHOLDS = {
    "min_episode_count",
    "max_episode_count",
    "min_action_density",
    "max_action_density",
    "min_activity_evenness",
    "max_activity_evenness",
    "min_action_density_first_half",
    "max_action_density_second_half",
    "min_non_technical_share",
    "max_non_technical_share",
}


def _empty_thresholds() -> dict[str, dict[str, float]]:
    return {k: {} for k in _KNOWN_STYLE_KEYS}


def load_style_thresholds(
    path: str | Path,
) -> tuple[dict[str, dict[str, float]], list[MarkovWarning]]:
    """Прочитать ``config/style_thresholds.yaml`` в нормализованную форму.

    Контракт:

    * формат — ``{thresholds: {<style>: {<rule>: <number>}}}`` либо
      сразу плоский ``{<style>: {...}}``;
    * неизвестные имена стилей → warning ``style_thresholds.unknown_style``,
      запись игнорируется;
    * неизвестные имена правил → warning
      ``style_thresholds.unknown_rule``, запись игнорируется;
    * при любых проблемах возвращается пустая таблица + warnings;
      классификатор спокойно деградирует в ``unclassified``.
    """

    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)

    warnings: list[MarkovWarning] = []
    if raw is None:
        return _empty_thresholds(), warnings
    if not isinstance(raw, dict):
        raise ValueError(
            f"Style thresholds config must be a YAML mapping, got: {type(raw).__name__}"
        )

    body: Any = (
        raw["thresholds"]
        if "thresholds" in raw and isinstance(raw["thresholds"], dict)
        else raw
    )

    if not isinstance(body, dict):
        warnings.append(
            MarkovWarning(
                code="style_thresholds.invalid_root",
                message="thresholds root is not a mapping; configuration ignored.",
            )
        )
        return _empty_thresholds(), warnings

    out: dict[str, dict[str, float]] = _empty_thresholds()
    for style_name, rules in body.items():
        if style_name not in _KNOWN_STYLE_KEYS:
            warnings.append(
                MarkovWarning(
                    code="style_thresholds.unknown_style",
                    message=f"Unknown style: {style_name!r}",
                    context={"style": str(style_name)},
                )
            )
            continue
        if not isinstance(rules, dict):
            warnings.append(
                MarkovWarning(
                    code="style_thresholds.invalid_rules",
                    message=f"Rules for style {style_name!r} must be a mapping.",
                    context={"style": str(style_name)},
                )
            )
            continue
        normalized: dict[str, float] = {}
        for rule_name, value in rules.items():
            if rule_name not in _KNOWN_THRESHOLDS:
                warnings.append(
                    MarkovWarning(
                        code="style_thresholds.unknown_rule",
                        message=(
                            f"Unknown rule {rule_name!r} for style "
                            f"{style_name!r}; ignored."
                        ),
                        context={"style": str(style_name), "rule": str(rule_name)},
                    )
                )
                continue
            try:
                normalized[str(rule_name)] = float(value)
            except (TypeError, ValueError):
                warnings.append(
                    MarkovWarning(
                        code="style_thresholds.invalid_value",
                        message=(
                            f"Threshold {style_name}.{rule_name} is not numeric"
                            f": {value!r}"
                        ),
                        context={"style": str(style_name), "rule": str(rule_name)},
                    )
                )
        out[str(style_name)] = normalized
    return out, warnings


def _matches_rules(metrics: EpisodeMetrics, rules: dict[str, float]) -> bool:
    """Все условия в ``rules`` должны быть выполнены, иначе False.

    Если требуется метрика, которой ``None`` (нет данных), правило не
    срабатывает: правильнее считать стиль ``unclassified``, чем
    выдумывать.
    """

    if not rules:
        return False

    accessors: dict[str, Any] = {
        "min_episode_count": metrics.episode_count,
        "max_episode_count": metrics.episode_count,
        "min_action_density": metrics.action_density,
        "max_action_density": metrics.action_density,
        "min_activity_evenness": metrics.activity_evenness,
        "max_activity_evenness": metrics.activity_evenness,
        "min_action_density_first_half": metrics.action_density_first_half,
        "max_action_density_second_half": metrics.action_density_second_half,
        "min_non_technical_share": metrics.non_technical_share,
        "max_non_technical_share": metrics.non_technical_share,
    }

    for rule_name, threshold in rules.items():
        actual = accessors.get(rule_name)
        if actual is None:
            return False
        if rule_name.startswith("min_") and not (actual >= threshold):
            return False
        if rule_name.startswith("max_") and not (actual <= threshold):
            return False
    return True


# Порядок проверки правил: «сначала более структурные, потом стрессовые».
_RULE_CHECK_ORDER: tuple[StyleLabel, ...] = (
    StyleLabel.BURNOUT,
    StyleLabel.SPEED_POWER,
    StyleLabel.ENDURANCE,
)


def classify_style(
    metrics: EpisodeMetrics,
    thresholds: dict[str, dict[str, float]] | None,
) -> tuple[StyleLabel, MarkovWarning | None]:
    """Описательная классификация стиля управления эпизодом.

    Возвращает кортеж ``(label, warning)``: если ни одно правило не
    сработало — ``(UNCLASSIFIED, MarkovWarning(code="style.no_rule_matched"))``.

    Порядок проверки фиксирован: ``burnout → speed_power → endurance``.
    Это нужно, чтобы более «специфическое» правило (с двумя порогами по
    половинам потока) проверялось раньше «менее специфического» правила
    с одним порогом по плотности.
    """

    if not thresholds:
        return StyleLabel.UNCLASSIFIED, MarkovWarning(
            code="style.no_rule_matched",
            message="Стилевые пороги не заданы; классификатор не выполнен.",
            context={"reason": "empty_thresholds"},
        )

    for label in _RULE_CHECK_ORDER:
        rules = thresholds.get(label.value, {})
        if _matches_rules(metrics, rules):
            return label, None

    return StyleLabel.UNCLASSIFIED, MarkovWarning(
        code="style.no_rule_matched",
        message=(
            "Ни одно правило стиля не сработало; стиль не определён."
            " См. config/style_thresholds.yaml."
        ),
        context={"athlete": metrics.athlete},
    )
