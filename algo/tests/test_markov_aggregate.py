"""Тесты агрегации индивидуальных Marков-моделей (TASK_SPEC_012).

Покрытие:

* агрегат — это ровно сумма ``transition_counts`` индивидуалов,
  построчно нормализованная;
* «посторонний» спортсмен (не из списка призёров) не попадает в
  агрегат — это проверяется тем, что результат для подмножества
  ``[A, B]`` отличается от результата для ``[A, B, C]``;
* инвариант ``sum(row) = 1 ± 1e-6`` сохраняется и для агрегата;
* пустой вход → ``aggregate.no_members`` warning, не падение.
"""

from __future__ import annotations

import numpy as np

from hpc_algo.markov_aggregate import fit_aggregate_markov
from hpc_algo.schema import (
    EpisodeState,
    MarkovIndividualResult,
)


def _indiv(
    athlete: str,
    counts: list[list[int]],
    visit: dict[str, int] | None = None,
) -> MarkovIndividualResult:
    states = [
        EpisodeState.MANOEUVRING,
        EpisodeState.GRIP,
        EpisodeState.OFF_BALANCE,
        EpisodeState.TECHNICAL_ACTION,
        EpisodeState.PAUSE,
    ]
    n = len(states)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        rs = sum(counts[i])
        if rs == 0:
            A[i, :] = 1.0 / n
        else:
            A[i] = np.array(counts[i], dtype=float) / rs
    return MarkovIndividualResult(
        athlete=athlete,
        mode="single",
        states=states,
        transition_counts=counts,
        transition_matrix=A.tolist(),
        stationary=[1.0 / n] * n,
        visit_counts=visit or {s.value: 0 for s in states},
        bout_count=1,
        episode_count=int(sum(sum(r) for r in counts)),
        warnings=[],
    )


def _zero_counts() -> list[list[int]]:
    return [[0] * 5 for _ in range(5)]


def test_fit_aggregate_sums_transition_counts() -> None:
    counts_a = _zero_counts()
    counts_a[0][1] = 5  # man → grip x5
    counts_a[1][3] = 2

    counts_b = _zero_counts()
    counts_b[0][1] = 3
    counts_b[2][3] = 4

    a = _indiv("A", counts_a, visit={"manoeuvring": 5, "grip": 2, "pause": 0,
                                     "off_balance": 0, "technical_action": 0})
    b = _indiv("B", counts_b, visit={"manoeuvring": 3, "off_balance": 4, "pause": 0,
                                     "grip": 0, "technical_action": 0})

    agg = fit_aggregate_markov([a, b], weight_class="48")

    assert agg.members == ["A", "B"]
    assert agg.members_count == 2
    # Сумма счётчиков = 5+3 в [0,1], 2 в [1,3], 4 в [2,3].
    assert agg.transition_counts[0][1] == 8
    assert agg.transition_counts[1][3] == 2
    assert agg.transition_counts[2][3] == 4

    # Visit_counts сложились.
    assert agg.visit_counts["manoeuvring"] == 8
    assert agg.visit_counts["grip"] == 2
    assert agg.visit_counts["off_balance"] == 4

    # Row-stochastic invariant.
    A = np.asarray(agg.transition_matrix)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)


def test_fit_aggregate_excludes_outsider_changes_result() -> None:
    """Если убрать одного индивидуала из входа — агрегат меняется.

    Это и есть инвариант «посторонний не попадает»: список агрегата
    управляется тем, кого мы передали, а не догадками внутри функции.
    """

    counts_a = _zero_counts()
    counts_a[0][1] = 1
    counts_b = _zero_counts()
    counts_b[1][2] = 1
    counts_c = _zero_counts()
    counts_c[3][4] = 1  # «посторонний»: только tech → pause

    a = _indiv("A", counts_a)
    b = _indiv("B", counts_b)
    c_outsider = _indiv("C", counts_c)

    agg_with_c = fit_aggregate_markov([a, b, c_outsider], weight_class="48")
    agg_without_c = fit_aggregate_markov([a, b], weight_class="48")

    # У «постороннего» c_outsider все переходы из tech → pause.
    # Без него строка tech → pause в агрегате должна быть нулевой
    # → row sum = 0 → uniform fill.
    A_with = np.asarray(agg_with_c.transition_matrix)
    A_without = np.asarray(agg_without_c.transition_matrix)
    assert not np.allclose(A_with[3], A_without[3])


def test_fit_aggregate_empty_emits_warning() -> None:
    agg = fit_aggregate_markov([], weight_class="60")
    codes = {w.code for w in agg.warnings}
    assert "aggregate.no_members" in codes
    A = np.asarray(agg.transition_matrix)
    # Все строки uniform (нет наблюдений).
    assert np.allclose(A, 1.0 / 5)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)


def test_fit_aggregate_warns_about_unobserved_rows() -> None:
    counts = _zero_counts()
    counts[0][1] = 1  # только переход из mano → grip
    a = _indiv("A", counts)
    agg = fit_aggregate_markov([a], weight_class="48")

    # 4 строки из 5 не наблюдались как «откуда».
    codes = [w.code for w in agg.warnings]
    assert codes.count("aggregate.unobserved_row") == 4


def test_fit_aggregate_preserves_alphabet_when_states_reordered() -> None:
    """Даже если индивидуал придёт с другим порядком states (мало ли
    что было в pickle), агрегат всегда живёт в каноническом 5-state
    алфавите — иначе divergence не даст согласованных результатов.
    """

    indiv_canonical = _indiv("A", _zero_counts())
    # подменим states на тот же набор, но в обратном порядке
    states_reversed = list(reversed(indiv_canonical.states))
    indiv_canonical.states[:] = states_reversed

    agg = fit_aggregate_markov([indiv_canonical], weight_class="48")
    assert [s.value for s in agg.states] == [
        "manoeuvring",
        "grip",
        "off_balance",
        "technical_action",
        "pause",
    ]
