"""Описательная сводка по спортсменам для тренерской вкладки.

Поверх ``frames`` из :func:`hpc_algo.mapping.load_mapped_sheets` и
:class:`ColumnMappingConfig` агрегируем для каждого человека (роль
``athlete``) число уникальных эпизодов (роль ``episode``).

Это **не** диагностика и не вывод о скрытых состояниях — задача модуля
дать тренеру наглядный список борцов, по которым в выгрузке вообще
есть данные. Все ограничения честно описываются в ``notes``.

Ключ эпизода — пара ``(sheet_name, str(episode_value))``: совпадающие
номера эпизодов в разных листах (например в разных весовых категориях)
не должны схлопываться между собой.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from hpc_algo.schema import (
    AthleteEpisodeStats,
    ColumnMappingConfig,
    HiddenGroup,
    TrainerAthleteSummary,
)


def _normalize_athlete(value: object) -> str | None:
    """Нормализовать значение колонки ``athlete``: trim и фильтр пустых."""

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _normalize_episode(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def build_athlete_episode_rollup(
    frames: dict[str, pd.DataFrame],
    config: ColumnMappingConfig,
) -> TrainerAthleteSummary:
    """Сводка по спортсменам и эпизодам.

    Возвращает :class:`TrainerAthleteSummary` всегда — даже если данных
    нет. ``athletes`` отсортирован по убыванию ``episode_count``,
    тай-брейк по ``athlete`` (стабильно).
    """

    notes: list[str] = []
    counts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    any_role_pair_seen = False

    for sheet_name, sheet_mapping in config.sheets.items():
        athlete_cols = sheet_mapping.roles.get(HiddenGroup.ATHLETE) or []
        episode_cols = sheet_mapping.roles.get(HiddenGroup.EPISODE) or []
        if not athlete_cols or not episode_cols:
            continue
        df = frames.get(sheet_name)
        if df is None or df.empty:
            continue

        athlete_col = next((c for c in athlete_cols if c in df.columns), None)
        episode_col = next((c for c in episode_cols if c in df.columns), None)
        if athlete_col is None or episode_col is None:
            notes.append(
                f"Лист «{sheet_name}»: колонки athlete/episode не найдены в данных."
            )
            continue

        any_role_pair_seen = True
        for athlete_raw, episode_raw in zip(
            df[athlete_col].tolist(), df[episode_col].tolist(), strict=False
        ):
            episode = _normalize_episode(episode_raw)
            if episode is None:
                continue
            athlete = _normalize_athlete(athlete_raw)
            if athlete is None:
                continue
            counts[athlete].add((sheet_name, episode))

    if not any_role_pair_seen:
        notes.append(
            "В column mapping не заданы одновременно роли athlete и episode "
            "ни на одном листе — сводка по спортсменам недоступна."
        )

    athletes = [
        AthleteEpisodeStats(athlete=name, episode_count=len(eps))
        for name, eps in counts.items()
    ]
    athletes.sort(key=lambda s: (-s.episode_count, s.athlete))

    total_episodes = sum(a.episode_count for a in athletes)

    return TrainerAthleteSummary(
        athletes=athletes,
        total_athletes=len(athletes),
        total_episodes=total_episodes,
        notes=notes,
    )
