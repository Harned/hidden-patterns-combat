"""Регрессии Honesty layer HMM-ветки.

Покрывают изменения, которые делают вывод HMM честным относительно
разрежённого ЗАП-сигнала и непрозрачности выбора варианта:

* low_zap_density guard и статус ``hmm_low_signal``;
* исключение all-noop серий из обучения и ``training_excluded_episodes``;
* posterior γ_t (T × K), confidence per-trajectory и average;
* tried_variants с пометкой applied/rejected и причинами.

Фикстуры — крошечные synthetic Excel-файлы (не зависят от реального
файла «Оценка СД содержание.xlsx»).
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.schema import AnalysisStatus

# ---------------------------------------------------------------------------
# Helpers / фикстуры
# ---------------------------------------------------------------------------


def _build_workbook(path: Path, rows: list[list[object]]) -> Path:
    """Собрать .xlsx с multi-row header и данными.

    Структура — сильно упрощённая копия `multirow_binary_zap_excel`:
    три строки заголовка + бинарная кодировка ЗАП по каналам.
    """

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "48"

    # row 0
    ws.append(
        [
            "ФИО борца",
            "Технико-тактический эпизод",
            None,
            "Завершающие атаку приемы (n)",
            None,
            None,
        ]
    )
    # row 1
    ws.append(
        [
            None,
            None,
            None,
            None,
            "Болевой прием",
            None,
        ]
    )
    # row 2
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Удержание",
            "На руку",
            "На ногу",
        ]
    )

    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


def _low_density_excel(tmp_path: Path) -> Path:
    """200 эпизодов / 8 ZAP-событий по 2 каналам: плотность 4% (< 5%).

    * min_episodes (30) — пройден (200 ≥ 30);
    * min_alphabet (2 после noop) — пройден (Удержание + На руку);
    * min_zap_events (8) — задаётся через AnalyzeConfig в тесте,
      по умолчанию 20 не подходит для разрежённой фикстуры;
    * median sequence length — 50 (4 борца по 50 строк) ≥ 2;
    * low_zap_density (5%) — ДОЛЖЕН СРАБОТАТЬ: 8/200 = 4%.
    """

    rows: list[list[object]] = []
    names = ["Иванов", "Петров", "Сидоров", "Кузнецов"]
    ep = 1
    for name in names:
        for k in range(50):
            # 8 событий: на двух каналах, на 0-й и 25-й позиции у каждого.
            udrzh = 1 if k == 0 else 0
            ruka = 1 if k == 25 else 0
            noga = 0
            rows.append([name, ep, 10 + k, udrzh, ruka, noga])
            ep += 1
    path = tmp_path / "low_density.xlsx"
    return _build_workbook(path, rows)


def _all_noop_per_athlete_excel(tmp_path: Path) -> Path:
    """Часть борцов имеет ВСЕ строки = noop, остальные — с ЗАП.

    Используется для проверки exclude-noop: должны попасть в
    trajectories с has_zap=False, но в обучении не участвовать.
    """

    rows: list[list[object]] = []
    # «Тихие» борцы — много строк без ЗАП.
    for name in ["Тихонов", "Молчанов"]:
        for k in range(15):
            rows.append([name, k + 1, 12, 0, 0, 0])
    # «Активные» борцы — с ЗАП-каналами.
    rng_pattern = [
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
    ]
    ep = 1
    for name in ["Иванов", "Петров", "Сидоров", "Кузнецов"]:
        for k, (a, b, c) in enumerate(rng_pattern):
            rows.append([name, ep + k, 12 + k, a, b, c])
        ep += len(rng_pattern)
    path = tmp_path / "all_noop_per_athlete.xlsx"
    return _build_workbook(path, rows)


# ---------------------------------------------------------------------------
# A1. low_zap_density guard
# ---------------------------------------------------------------------------


def test_low_zap_density_emits_warning_and_sets_low_signal_status(
    tmp_path: Path,
) -> None:
    src = _low_density_excel(tmp_path)
    cfg = preflight_mapping(src)
    # min_zap_events по умолчанию = 20; на разрежённой фикстуре их 8.
    # Ослабляем, чтобы дойти до low_zap_density.
    result = analyze_source(
        src,
        AnalyzeConfig(column_mapping=cfg, hmm_min_zap_events=5),
    )

    codes = {w.code for w in result.warnings}
    assert "hmm.low_zap_density" in codes, codes
    # Это soft-guard: HMM всё равно обучилась, статус — hmm_low_signal.
    assert result.status == AnalysisStatus.HMM_LOW_SIGNAL
    assert result.hmm is not None

    # В контексте warning есть полезные числа.
    soft = next(w for w in result.warnings if w.code == "hmm.low_zap_density")
    ctx = soft.context
    assert ctx["non_noop_steps"] >= 1
    assert ctx["total_steps"] >= ctx["non_noop_steps"]
    assert ctx["density"] < ctx["threshold"]


# ---------------------------------------------------------------------------
# A2. exclude-noop в обучении
# ---------------------------------------------------------------------------


def test_all_noop_sequences_are_excluded_from_training(tmp_path: Path) -> None:
    src = _all_noop_per_athlete_excel(tmp_path)
    cfg = preflight_mapping(src)
    result = analyze_source(src, AnalyzeConfig(column_mapping=cfg))

    assert result.hmm is not None, result.warnings
    hmm = result.hmm
    # 2 «тихих» борца -> 2 серии без ZAP -> ровно 2 исключённые серии.
    assert hmm.training_excluded_episodes == 2
    assert hmm.no_zap_trajectories == 2

    no_zap_traj = [tr for tr in hmm.trajectories if not tr.has_zap]
    assert len(no_zap_traj) == 2
    for tr in no_zap_traj:
        assert all(t == "_noop_" for t in tr.observation_tokens)
        # Длина траектории сохранена — она нужна для отображения.
        assert tr.length > 0


# ---------------------------------------------------------------------------
# B1. gamma posterior + confidence
# ---------------------------------------------------------------------------


def test_episode_traces_carry_state_posterior_and_confidence(
    dense_hmm_ready_excel: Path,
) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    result = analyze_source(
        dense_hmm_ready_excel, AnalyzeConfig(column_mapping=cfg)
    )
    assert result.hmm is not None
    hmm = result.hmm

    # На honest пути posterior считается через predict_proba (categorical
    # ветка). Проверяем форму и нормализацию.
    n_states = hmm.parameters.n_states
    seen = 0
    for tr in hmm.trajectories:
        if tr.state_posterior is None:
            continue
        seen += 1
        # T × K
        assert len(tr.state_posterior) == tr.length
        for row in tr.state_posterior:
            assert len(row) == n_states
            assert abs(sum(row) - 1.0) < 1e-3
        assert tr.confidence is not None
        assert 0.0 <= tr.confidence <= 1.0
    assert seen > 0, "ожидаем хотя бы один эпизод с заполненным state_posterior"
    assert hmm.average_confidence is not None
    assert 0.0 <= hmm.average_confidence <= 1.0


# ---------------------------------------------------------------------------
# C1. tried_variants
# ---------------------------------------------------------------------------


def test_hmm_result_records_variant_attempts(
    dense_hmm_ready_excel: Path,
) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    result = analyze_source(
        dense_hmm_ready_excel, AnalyzeConfig(column_mapping=cfg)
    )
    assert result.hmm is not None

    attempts = result.hmm.tried_variants
    assert attempts, "tried_variants должен быть заполнен"

    # Ровно один applied.
    applied = [a for a in attempts if a.status == "applied"]
    assert len(applied) == 1
    assert applied[0].variant == result.hmm.parameters.variant

    # Detailed на маленьком dense — он либо не запущен (rejected_by_guard),
    # либо отбракован по sanity/BIC. В любом случае присутствует попытка
    # с непустой причиной.
    detailed = [a for a in attempts if a.variant == "detailed_7state"]
    if detailed:
        assert detailed[0].status != "applied" or applied[0].variant == "detailed_7state"
        if detailed[0].status != "applied":
            assert detailed[0].reason
