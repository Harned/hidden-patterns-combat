"""Индивидуальная наблюдаемая 5-state Marков-цепь по эпизодам (TASK_SPEC_011).

Контракт модуля:

* вход — список :class:`RawEpisode` (см. :mod:`hpc_algo.episode_split`)
  и :class:`StateGroupsConfig` (см. :mod:`hpc_algo.state_groups`);
* выход — :class:`MarkovIndividualResult` для одного спортсмена,
  содержащий ``transition_counts``, ``transition_matrix``,
  ``stationary``, ``visit_counts`` и список :class:`MarkovWarning`.

Транзиции считаются ВНУТРИ bout'а — на границе ``bout_id`` пары
``(prev, next)`` не учитываются. Это соответствует требованию
``TASK_SPEC_012`` («не индуцировать ложные переходы между разными
поединками») и применимо уже на уровне индивидуальной модели.

Стационарка ``π`` — собственный вектор ``A^T`` для собственного
значения ≈ 1. При недостатке данных (≥ 2 состояний без посещений)
включается fallback на time-averaging ``visit_counts`` и эмитится
``MarkovWarning(code="mc.non_ergodic")``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from hpc_algo.episode_split import RawEpisode
from hpc_algo.schema import (
    EpisodeRecord,
    EpisodeState,
    MarkovIndividualResult,
    MarkovMode,
    MarkovWarning,
)
from hpc_algo.state_groups import StateGroupsConfig

_MULTI_ORDER: tuple[EpisodeState, ...] = (
    EpisodeState.MANOEUVRING,
    EpisodeState.GRIP,
    EpisodeState.OFF_BALANCE,
    EpisodeState.TECHNICAL_ACTION,
)

_DEFAULT_PRIORITY: tuple[EpisodeState, ...] = (
    EpisodeState.TECHNICAL_ACTION,
    EpisodeState.OFF_BALANCE,
    EpisodeState.GRIP,
    EpisodeState.MANOEUVRING,
    EpisodeState.PAUSE,
)

_ALPHABET: tuple[EpisodeState, ...] = (
    EpisodeState.MANOEUVRING,
    EpisodeState.GRIP,
    EpisodeState.OFF_BALANCE,
    EpisodeState.TECHNICAL_ACTION,
    EpisodeState.PAUSE,
)


def _resolve_priority(cfg: StateGroupsConfig) -> tuple[EpisodeState, ...]:
    """Привести priority из YAML к кортежу EpisodeState; неизвестные — отфильтровать."""

    out: list[EpisodeState] = []
    for name in cfg.priority:
        try:
            out.append(EpisodeState(name))
        except ValueError:
            continue
    if not out:
        return _DEFAULT_PRIORITY
    return tuple(out)


def _episode_active_states(
    feature_values: dict[str, float],
    cfg: StateGroupsConfig,
) -> set[EpisodeState]:
    """Какие группы (без pause) имеют ненулевую активность в эпизоде.

    Колонки YAML, не существующие в листе, при чтении значений дают 0
    (см. :func:`hpc_algo.episode_split._normalize_feature_value` /
    :func:`pandas.Series.get`), поэтому отдельную обработку здесь не
    делаем — отсутствующая колонка просто не добавляет активности.
    """

    active: set[EpisodeState] = set()
    for state_name, columns in cfg.states.items():
        if state_name == EpisodeState.PAUSE.value:
            continue
        try:
            state = EpisodeState(state_name)
        except ValueError:
            continue
        for col in columns:
            if feature_values.get(col, 0.0) > 0:
                active.add(state)
                break
    return active


def assign_state_single(
    feature_values: dict[str, float],
    cfg: StateGroupsConfig,
) -> EpisodeState:
    """Назначить эпизоду одно состояние по приоритету (`single`-режим)."""

    active = _episode_active_states(feature_values, cfg)
    for s in _resolve_priority(cfg):
        if s == EpisodeState.PAUSE:
            continue
        if s in active:
            return s
    return EpisodeState.PAUSE


def assign_state_multi(
    feature_values: dict[str, float],
    cfg: StateGroupsConfig,
) -> list[EpisodeState]:
    """Развернуть эпизод в подпоследовательность активных групп (`multi`-режим)."""

    active = _episode_active_states(feature_values, cfg)
    seq = [s for s in _MULTI_ORDER if s in active]
    if not seq:
        return [EpisodeState.PAUSE]
    return seq


def build_episode_sequence(
    raw_episodes: Iterable[RawEpisode],
    cfg: StateGroupsConfig,
) -> list[EpisodeRecord]:
    """Построить список :class:`EpisodeRecord` из «сырых» эпизодов."""

    records: list[EpisodeRecord] = []
    for raw in raw_episodes:
        if cfg.mode == "multi":
            state: EpisodeState | list[EpisodeState] = assign_state_multi(
                raw.feature_values, cfg
            )
        else:
            state = assign_state_single(raw.feature_values, cfg)
        records.append(
            EpisodeRecord(
                athlete=raw.athlete,
                bout_id=raw.bout_id,
                episode_idx=raw.episode_idx_in_bout,
                episode_duration=raw.episode_time,
                pause_duration=raw.pause_time,
                score=raw.score,
                state=state,
            )
        )
    return records


def _flatten_record_states(rec: EpisodeRecord) -> list[EpisodeState]:
    if isinstance(rec.state, list):
        return list(rec.state)
    return [rec.state]


def _stationary_distribution(
    A: np.ndarray,
    visit: dict[str, int],
) -> tuple[np.ndarray, MarkovWarning | None]:
    """Стационарка через ev(A^T) ≈ 1; fallback на time-averaging visit_counts.

    Решение «когда fallback» — эвристическое: если **≥ 2 состояний не
    наблюдались**, ev-метод склонен возвращать вырожденный вектор. В
    таком случае честнее показать частоты, чем экстраполировать.
    """

    n = A.shape[0]
    unobserved = sum(1 for s in _ALPHABET if visit.get(s.value, 0) == 0)

    def _time_average() -> np.ndarray:
        total = sum(visit.values())
        if total == 0:
            return np.full(n, 1.0 / n)
        return np.array([visit.get(s.value, 0) for s in _ALPHABET], dtype=float) / total

    if unobserved >= 2:
        return _time_average(), MarkovWarning(
            code="mc.non_ergodic",
            message=(
                "Цепь неэргодична или данные разрежены"
                f" ({unobserved} из {n} состояний без посещений);"
                " π оценено через time-averaging visit_counts."
            ),
            context={"unobserved": int(unobserved)},
        )

    try:
        eigvals, eigvecs = np.linalg.eig(A.T)
    except np.linalg.LinAlgError:
        return _time_average(), MarkovWarning(
            code="mc.non_ergodic",
            message="Не удалось вычислить ev(A^T); fallback на time-averaging.",
        )

    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    v = np.real(eigvecs[:, idx])
    s = v.sum()
    if s == 0 or not np.isfinite(s):
        return _time_average(), MarkovWarning(
            code="mc.non_ergodic",
            message="Сумма ev-вектора нулевая; fallback на time-averaging.",
        )
    if s < 0:
        v = -v
        s = -s
    v = np.clip(v, 0.0, None)
    s = v.sum()
    if s == 0:
        return _time_average(), MarkovWarning(
            code="mc.non_ergodic",
            message="После клипа ev-вектор обнулился; fallback на time-averaging.",
        )
    return v / s, None


def fit_individual_markov(
    athlete: str,
    records: Iterable[EpisodeRecord],
    mode: MarkovMode = "single",
) -> MarkovIndividualResult:
    """Построить индивидуальную наблюдаемую 5-state Marков-цепь.

    Параметры:

    * ``athlete`` — точное значение в ``EpisodeRecord.athlete``, по
      которому фильтруем.
    * ``records`` — все эпизоды любого числа спортсменов; функция сама
      возьмёт только записи нужного.
    * ``mode`` — для метаданных результата; реальный режим (`single`/
      `multi`) уже определён при ``build_episode_sequence``.
    """

    state_to_idx = {s: i for i, s in enumerate(_ALPHABET)}
    n = len(_ALPHABET)
    counts = np.zeros((n, n), dtype=np.int64)
    visit: dict[str, int] = {s.value: 0 for s in _ALPHABET}
    warnings: list[MarkovWarning] = []

    bouts: dict[str, list[EpisodeState]] = {}
    bout_order: list[str] = []
    episode_count_for_athlete = 0

    for rec in records:
        if rec.athlete != athlete:
            continue
        if rec.bout_id not in bouts:
            bouts[rec.bout_id] = []
            bout_order.append(rec.bout_id)
        bouts[rec.bout_id].extend(_flatten_record_states(rec))
        episode_count_for_athlete += 1

    if not bout_order:
        warnings.append(
            MarkovWarning(
                code="mc.no_episodes",
                message=f"Спортсмен '{athlete}' не имеет ни одного эпизода в выборке.",
                context={"athlete": athlete},
            )
        )

    for bid in bout_order:
        seq = bouts[bid]
        for state in seq:
            visit[state.value] += 1
        for prev_state, next_state in zip(seq, seq[1:], strict=False):
            counts[state_to_idx[prev_state], state_to_idx[next_state]] += 1

    A = np.zeros_like(counts, dtype=float)
    for i in range(n):
        row_sum = counts[i].sum()
        if row_sum == 0:
            A[i, :] = 1.0 / n
            warnings.append(
                MarkovWarning(
                    code="mc.unobserved_row",
                    message=(
                        f"Состояние '{_ALPHABET[i].value}' не наблюдалось как 'откуда';"
                        " строка матрицы переходов заполнена равномерно."
                    ),
                    context={"state": _ALPHABET[i].value},
                )
            )
        else:
            A[i] = counts[i] / row_sum

    pi, pi_warning = _stationary_distribution(A, visit)
    if pi_warning is not None:
        warnings.append(pi_warning)

    row_sums = A.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        warnings.append(
            MarkovWarning(
                code="mc.row_sum_violation",
                message="Сумма строки A отклоняется от 1 более чем на 1e-6.",
                context={"row_sums": [float(x) for x in row_sums]},
            )
        )

    return MarkovIndividualResult(
        athlete=athlete,
        mode=mode,
        states=list(_ALPHABET),
        transition_counts=[[int(x) for x in row] for row in counts],
        transition_matrix=[[float(x) for x in row] for row in A],
        stationary=[float(x) for x in pi],
        visit_counts=visit,
        bout_count=len(bout_order),
        episode_count=episode_count_for_athlete,
        warnings=warnings,
    )
