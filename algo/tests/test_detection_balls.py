"""Регрессии для эвристики детекции «Баллы» как ZAP-кандидата.

В реальных файлах вида ``docs/Оценка СД содержание.xlsx`` числовая
колонка «Баллы» — это общая судейская оценка эпизода. По
``DOMAIN_SPEC.md`` она является фактом ZAP-наблюдения (ненулевой
балл => зафиксирован ЗАП). Алгоритм при этом не интерпретирует
конкретные числа как разные ЗАП-классы.

До правки эвристики `algo/hpc_algo/detection.py` маркер «балл»/«оценк»
не распознавался, и колонка не подсвечивалась как ZAP-кандидат —
в итоге плотность ZAP-сигнала оставалась низкой, и срабатывал
``hmm.low_zap_density``. Эти тесты фиксируют поведение после правки.
"""

from __future__ import annotations

from pathlib import Path

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.detection import detect_columns
from hpc_algo.loading import load_excel
from hpc_algo.mapping import read_sheet_with_header_rows
from hpc_algo.schema import HiddenGroup, SheetMapping


def test_balls_supercol_detected_as_zap_candidate(balls_zap_excel: Path) -> None:
    """``detect_columns`` подсвечивает «Баллы» как ZAP-кандидата.

    Проверяем сам факт детекции и то, что score остаётся ниже
    верхней «уверенной» границы (≤ 0.85), чтобы кандидат не выглядел
    как полноценная категориальная ЗАП-метка без явного подтверждения,
    как и предписывает honesty layer.
    """

    loaded = load_excel(balls_zap_excel)
    report = detect_columns(loaded)

    balls_candidates = [
        c
        for c in report.candidates
        if c.group == HiddenGroup.ZAP and "Баллы" in c.column
    ]
    assert balls_candidates, (
        "Ожидаем ZAP-кандидата для колонки «Баллы», но эвристика его не вернула. "
        f"Все кандидаты: {[(c.group.value, c.column, c.score) for c in report.candidates]}"
    )
    candidate = balls_candidates[0]
    assert 0.4 <= candidate.score < 1.0
    # На синтетике плотность баллов выше реальной (35% vs ~12%) -> поддержка
    # выше, поэтому дополнительно требуем, чтобы score не превышал
    # 0.85 (нижняя граница уверенной ЗАП-метки), иначе мы рискуем
    # заявить категориальный ЗАП там, где его нет.
    assert candidate.score <= 0.85


def test_balls_in_mapping_lifts_zap_density(balls_zap_excel: Path) -> None:
    """С «Баллы» в роли ZAP плотность сигнала достаточна для HMM.

    Сценарий повторяет реальный кейс: оператор вручную добавляет
    «Баллы» в роль ``ZAP``. Без этой правки эвристики удержание/«на
    руку» дают единичные ЗАП-события и `hmm.low_zap_density` срабатывает.
    """

    cfg = preflight_mapping(balls_zap_excel)
    sheet_name = next(iter(cfg.sheets))
    sheet_cfg = cfg.sheets[sheet_name]
    df = read_sheet_with_header_rows(
        balls_zap_excel, sheet_name, sheet_cfg.header_rows
    )
    balls_col = next(c for c in df.columns if "Баллы" in str(c))
    new_roles = {role: list(cols) for role, cols in sheet_cfg.roles.items()}
    new_roles.setdefault(HiddenGroup.ZAP, [])
    if balls_col not in new_roles[HiddenGroup.ZAP]:
        new_roles[HiddenGroup.ZAP].append(balls_col)
    cfg.sheets[sheet_name] = SheetMapping(
        header_rows=sheet_cfg.header_rows,
        data_start_row=sheet_cfg.data_start_row,
        roles=new_roles,
    )

    result = analyze_source(balls_zap_excel, AnalyzeConfig(column_mapping=cfg))
    assert result.hmm is not None, [w.code for w in result.warnings]

    codes = {w.code for w in result.warnings}
    assert "hmm.low_zap_density" not in codes, (
        f"Ожидаем достаточную плотность ZAP, но guard сработал. Warnings: {codes}"
    )

    # «Баллы» как count-канал: токен берётся из имени канала.
    alphabet = result.hmm.parameters.observation_labels
    assert any("Балл" in tok for tok in alphabet), alphabet
