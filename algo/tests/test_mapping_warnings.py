"""Регрессии для проактивных warnings мэппинга (Phase 3).

* ``mapping.zap_candidate_unmapped`` — оператор пропустил колонку,
  которую эвристика уверенно подсветила как ZAP-кандидата.
* ``mapping.bout_missing`` — на листе нет роли ``bout`` и виртуальный
  bout восстановить не удалось (нет athlete/episode или sanity-guard
  отбраковал).

Цель — не заменять собой DOMAIN_SPEC, а делать недостатки разметки
явными и выводить пользователя на нужный шаг в UI.
"""

from __future__ import annotations

from pathlib import Path

from hpc_algo.api import AnalyzeConfig, analyze_source
from hpc_algo.schema import (
    ColumnMappingConfig,
    HiddenGroup,
    SheetMapping,
)


def test_zap_candidate_unmapped_fires_when_balls_not_in_zap_role(
    balls_zap_excel: Path,
) -> None:
    """Оператор разметил athlete/episode, но не положил «Баллы» в ZAP."""

    cfg = ColumnMappingConfig(
        sheets={
            "Общее": SheetMapping(
                header_rows=[0, 1, 2],
                roles={
                    HiddenGroup.ATHLETE: ["ФИО борца"],
                    HiddenGroup.EPISODE: [
                        "Технико-тактический эпизод | № эпизода"
                    ],
                },
            )
        }
    )
    result = analyze_source(balls_zap_excel, AnalyzeConfig(column_mapping=cfg))
    unmapped = [
        w for w in result.warnings
        if w.code == "mapping.zap_candidate_unmapped"
    ]
    assert any("Балл" in w.context.get("column", "") for w in unmapped), (
        "Ожидаем warning с колонкой «Баллы», получено: "
        f"{[(w.context.get('column'), w.context.get('score')) for w in unmapped]}"
    )


def test_zap_candidate_unmapped_silenced_when_balls_explicitly_zap(
    balls_zap_excel: Path,
) -> None:
    """Если «Баллы» в роли ZAP, warning не эмитится."""

    cfg = ColumnMappingConfig(
        sheets={
            "Общее": SheetMapping(
                header_rows=[0, 1, 2],
                roles={
                    HiddenGroup.ATHLETE: ["ФИО борца"],
                    HiddenGroup.EPISODE: [
                        "Технико-тактический эпизод | № эпизода"
                    ],
                    HiddenGroup.ZAP: ["Баллы"],
                },
            )
        }
    )
    result = analyze_source(balls_zap_excel, AnalyzeConfig(column_mapping=cfg))
    codes = {w.code for w in result.warnings}
    assert "mapping.zap_candidate_unmapped" not in codes


def test_bout_missing_fires_when_athlete_role_absent(
    bout_resets_excel: Path,
) -> None:
    """Без athlete роли виртуальный bout не восстановить → mapping.bout_missing."""

    cfg = ColumnMappingConfig(
        sheets={
            "48": SheetMapping(
                header_rows=[0, 1, 2],
                roles={
                    HiddenGroup.EPISODE: [
                        "Технико-тактический эпизод | № эпизода"
                    ],
                    HiddenGroup.ZAP: [
                        "Завершающие атаку приемы (n) | Удержание"
                    ],
                },
            )
        }
    )
    result = analyze_source(bout_resets_excel, AnalyzeConfig(column_mapping=cfg))
    codes = {w.code for w in result.warnings}
    assert "mapping.bout_missing" in codes
    assert "hmm.bout_inferred_from_episode_resets" not in codes
