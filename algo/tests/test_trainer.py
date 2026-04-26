from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.mapping import load_mapped_sheets
from hpc_algo.schema import ColumnMappingConfig, HiddenGroup, SheetMapping
from hpc_algo.trainer import build_athlete_episode_rollup


def test_trainer_rollup_counts_unique_episodes_per_athlete(
    multirow_header_excel: Path,
) -> None:
    cfg = preflight_mapping(multirow_header_excel)
    frames = load_mapped_sheets(multirow_header_excel, cfg)

    summary = build_athlete_episode_rollup(frames, cfg)

    by_name = {a.athlete: a.episode_count for a in summary.athletes}
    assert by_name == {"Иванов": 2, "Петров": 2, "Сидоров": 1}

    # Сортировка: по убыванию episode_count, тай-брейк по имени (стабильно).
    assert [a.athlete for a in summary.athletes] == [
        "Иванов",
        "Петров",
        "Сидоров",
    ]
    assert summary.total_athletes == 3
    assert summary.total_episodes == 5
    assert summary.notes == []


def test_trainer_rollup_returns_note_when_roles_missing(tmp_path: Path) -> None:
    path = tmp_path / "no_roles.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(path, index=False)
    cfg = ColumnMappingConfig(
        sheets={"Sheet1": SheetMapping(header_rows=[0], roles={})}
    )
    frames = load_mapped_sheets(path, cfg)

    summary = build_athlete_episode_rollup(frames, cfg)

    assert summary.athletes == []
    assert summary.total_athletes == 0
    assert summary.total_episodes == 0
    assert summary.notes  # явная нота вместо тихого пустого ответа


def test_trainer_rollup_skips_rows_with_empty_episode(tmp_path: Path) -> None:
    path = tmp_path / "with_gaps.xlsx"
    pd.DataFrame(
        {
            "ФИО": ["Иванов", "Иванов", "Петров", "Петров"],
            "Эпизод": [1, None, 1, 2],
            "ЗАП": ["ЗАП-Р", "ЗАП-Т", "удержание", "ЗАП-Н"],
        }
    ).to_excel(path, index=False)
    cfg = ColumnMappingConfig(
        sheets={
            "Sheet1": SheetMapping(
                header_rows=[0],
                roles={
                    HiddenGroup.ATHLETE: ["ФИО"],
                    HiddenGroup.EPISODE: ["Эпизод"],
                },
            )
        }
    )
    frames = load_mapped_sheets(path, cfg)

    summary = build_athlete_episode_rollup(frames, cfg)

    by_name = {a.athlete: a.episode_count for a in summary.athletes}
    assert by_name == {"Иванов": 1, "Петров": 2}


def test_analyze_source_includes_trainer_summary(
    multirow_header_excel: Path,
) -> None:
    cfg = preflight_mapping(multirow_header_excel)
    result = analyze_source(
        multirow_header_excel, AnalyzeConfig(column_mapping=cfg)
    )

    assert result.trainer_athlete_summary is not None
    assert result.trainer_athlete_summary.total_athletes == 3
