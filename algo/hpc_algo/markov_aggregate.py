"""Агрегатная 5-state Marков-цепь по призёрам 1–3 (TASK_SPEC_012).

Контракт: на вход — список :class:`MarkovIndividualResult` для одной
весовой категории. Считаем суммы ``transition_counts`` и
``visit_counts`` (никакой склейки последовательностей — это
исключило бы ложные переходы между разными атлетами на стыке).
Дальше нормализуем по строкам и переиспользуем стационарку из
:mod:`hpc_algo.markov_individual`.

Никаких HMM-полей, никакого `fighter_style`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

# `_stationary_distribution` помечен как module-private, но используется
# здесь намеренно: алгоритм идентичен индивидуальному (тот же 5-state
# алфавит, тот же подход с fallback на time-averaging). Дублировать
# реализацию не хочу — это разошлось бы со временем. Если в будущем
# появится третий клиент, вынесем функцию в `markov_common.py`.
from hpc_algo.markov_individual import _ALPHABET, _stationary_distribution
from hpc_algo.schema import (
    EpisodeState,
    MarkovAggregateResult,
    MarkovIndividualResult,
    MarkovMode,
    MarkovWarning,
)


def fit_aggregate_markov(
    individuals: Iterable[MarkovIndividualResult],
    weight_class: str,
    *,
    mode: MarkovMode = "single",
) -> MarkovAggregateResult:
    """Собрать агрегатную модель из индивидуалов.

    Если на вход не пришло ни одного спортсмена — возвращается пустой
    результат с warning ``aggregate.no_members``: оркестратору проще
    решить, что делать (отчёт по пустой категории, либо пропустить).
    """

    members: list[MarkovIndividualResult] = list(individuals)
    n = len(_ALPHABET)
    counts = np.zeros((n, n), dtype=np.int64)
    visit: dict[str, int] = {s.value: 0 for s in _ALPHABET}
    warnings: list[MarkovWarning] = []
    bout_total = 0
    episode_total = 0

    if not members:
        warnings.append(
            MarkovWarning(
                code="aggregate.no_members",
                message=f"Категория '{weight_class}' не имеет ни одного спортсмена.",
                context={"weight_class": weight_class},
            )
        )

    for indiv in members:
        # Защита: на случай, если states у индивидуала вдруг другой алфавит —
        # выравниваем по позиции state.value, чтобы агрегат всегда жил
        # в каноническом порядке `_ALPHABET`.
        idx_map = _index_map(indiv.states)
        for i, src in enumerate(indiv.states):
            si = idx_map.get(src.value)
            if si is None:
                continue
            visit[src.value] += int(indiv.visit_counts.get(src.value, 0))
            for j, dst in enumerate(indiv.states):
                dj = idx_map.get(dst.value)
                if dj is None:
                    continue
                counts[si, dj] += int(indiv.transition_counts[i][j])
        bout_total += int(indiv.bout_count)
        episode_total += int(indiv.episode_count)

    A = np.zeros_like(counts, dtype=float)
    for i in range(n):
        row_sum = counts[i].sum()
        if row_sum == 0:
            A[i, :] = 1.0 / n
            warnings.append(
                MarkovWarning(
                    code="aggregate.unobserved_row",
                    message=(
                        f"Состояние '{_ALPHABET[i].value}' не наблюдалось как 'откуда'"
                        " ни у одного из агрегируемых спортсменов; строка"
                        " матрицы заполнена равномерно."
                    ),
                    context={
                        "state": _ALPHABET[i].value,
                        "weight_class": weight_class,
                    },
                )
            )
        else:
            A[i] = counts[i] / row_sum

    pi, pi_warning = _stationary_distribution(A, visit)
    if pi_warning is not None:
        # переиспользуем тот же warning-код, чтобы UI не приходилось
        # различать «индивидуальную» и «агрегатную» неэргодичность —
        # это семантически одно и то же.
        warnings.append(pi_warning)

    row_sums = A.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        warnings.append(
            MarkovWarning(
                code="aggregate.row_sum_violation",
                message="Сумма строки A_aggregate отклоняется от 1 более чем на 1e-6.",
                context={"row_sums": [float(x) for x in row_sums]},
            )
        )

    return MarkovAggregateResult(
        weight_class=weight_class,
        members=[m.athlete for m in members],
        members_count=len(members),
        mode=mode,
        states=list(_ALPHABET),
        transition_counts=[[int(x) for x in row] for row in counts],
        transition_matrix=[[float(x) for x in row] for row in A],
        stationary=[float(x) for x in pi],
        visit_counts=visit,
        bout_count_total=bout_total,
        episode_count_total=episode_total,
        warnings=warnings,
    )


def _index_map(states: list[EpisodeState]) -> dict[str, int]:
    """Соответствие state.value → его индекс в каноническом ``_ALPHABET``.

    Сделано отдельной функцией, чтобы независимо тестировать.
    """

    canonical = {s.value: i for i, s in enumerate(_ALPHABET)}
    out: dict[str, int] = {}
    for s in states:
        if s.value in canonical:
            out[s.value] = canonical[s.value]
    return out
