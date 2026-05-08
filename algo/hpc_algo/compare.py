"""Дивергенции и ранжирование индивидуалов относительно агрегата (TASK_SPEC_012).

Метрики:

* ``kl_transitions`` — взвешенное среднее KL-дивергенций по строкам
  ``A_individual`` относительно ``A_aggregate``, веса — ``π_aggregate``.
  Эпсилон-сглаживание агрегата (1e-12) применяется только при подсчёте
  KL, чтобы вырожденные строки не приводили к ``+inf``.
* ``l1_stationary`` — L1-расстояние между стационарными распределениями.
* ``composite = α · l1_stationary + (1 − α) · kl_transitions`` —
  по `TASK_SPEC_012` § Артефакты, ``α`` по умолчанию 0.5.

KL — асимметричная (это намеренно: нам важно «насколько индивидуальные
переходы отклоняются от агрегатных», а не наоборот). L1 — симметричная.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from hpc_algo.schema import (
    Divergence,
    MarkovAggregateResult,
    MarkovIndividualResult,
    RankingEntry,
)

_KL_EPS = 1e-12


def _kl_row(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) c эпсилон-сглаживанием q.

    Если p[k] = 0 — слагаемое 0 (по соглашению 0·log 0 = 0).
    Если q[k] = 0 — заменяем на ``_KL_EPS`` (а не на полную равномерность),
    чтобы агрегат с «дырой» строго наказывался индивидуалом, который
    туда заходит.
    """

    q_safe = np.maximum(q, _KL_EPS)
    p_pos = p > 0
    if not np.any(p_pos):
        return 0.0
    return float(np.sum(p[p_pos] * np.log(p[p_pos] / q_safe[p_pos])))


def divergence(
    individual: MarkovIndividualResult,
    aggregate: MarkovAggregateResult,
    *,
    alpha: float = 0.5,
) -> Divergence:
    """Подсчитать KL по транзициям + L1 по стационарке + композит."""

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    A_indiv = np.asarray(individual.transition_matrix, dtype=float)
    A_agg = np.asarray(aggregate.transition_matrix, dtype=float)
    pi_agg = np.asarray(aggregate.stationary, dtype=float)

    if A_indiv.shape != A_agg.shape:
        raise ValueError(
            f"Shape mismatch: individual {A_indiv.shape} vs aggregate {A_agg.shape}"
        )

    n = A_indiv.shape[0]
    kl_total = 0.0
    for i in range(n):
        kl_total += float(pi_agg[i]) * _kl_row(A_indiv[i], A_agg[i])
    # numerical jitter может дать чуть ниже нуля при ε-сглаживании;
    # клипаем — KL по построению неотрицательна.
    kl_total = float(max(0.0, kl_total))

    pi_indiv = np.asarray(individual.stationary, dtype=float)
    l1 = float(np.sum(np.abs(pi_indiv - pi_agg)))

    composite = float(alpha * l1 + (1.0 - alpha) * kl_total)

    return Divergence(
        individual=individual.athlete,
        weight_class=aggregate.weight_class,
        kl_transitions=kl_total,
        l1_stationary=l1,
        composite=composite,
        alpha=alpha,
    )


def rank_within_class(
    individuals: Iterable[MarkovIndividualResult],
    aggregate: MarkovAggregateResult,
    *,
    alpha: float = 0.5,
    places: dict[str, int] | None = None,
) -> list[RankingEntry]:
    """Отсортировать индивидуалов по возрастанию композитной метрики.

    ``places`` — необязательный словарь ``athlete -> 1/2/3`` для
    отображения в отчёте; на ранжирование не влияет (порядок задаётся
    composite-метрикой). При равенстве composite порядок стабилен:
    сортировка устойчива по возрастанию `(composite, athlete)`.
    """

    items = list(individuals)
    divs = [divergence(ind, aggregate, alpha=alpha) for ind in items]

    paired = sorted(
        zip(items, divs, strict=True),
        key=lambda pair: (pair[1].composite, pair[0].athlete),
    )

    out: list[RankingEntry] = []
    for rank, (ind, d) in enumerate(paired, start=1):
        place = None
        if places is not None:
            place = places.get(ind.athlete)
        out.append(
            RankingEntry(
                athlete=ind.athlete,
                weight_class=aggregate.weight_class,
                place=place,
                divergence=d,
                rank=rank,
            )
        )
    return out
