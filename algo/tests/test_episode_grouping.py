"""Регрессии на составной ключ эпизода.

В реальных файлах "Оценка СД" нумерация эпизодов перезапускается
у каждого борца/в каждой схватке. Простой ``nunique`` по колонке
``№ эпизода`` систематически занижал счёт, а HMM склеивала строки
разных борцов в одну последовательность. Тесты ниже фиксируют, что
теперь группировка идёт по составному ключу
``athlete + bout + episode``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.baseline import build_baseline_with_mapping
from hpc_algo.hmm import build_observation_sequences
from hpc_algo.mapping import (
    episode_grouping_columns,
    episode_key,
    load_mapped_sheets,
)
from hpc_algo.schema import (
    AuditReport,
    ColumnMappingConfig,
    HiddenGroup,
    SheetMapping,
)


def _empty_audit() -> AuditReport:
    return AuditReport(
        sheets=[], total_rows=0, total_cells=0, overall_null_ratio=0.0
    )


def _two_athletes_same_numbers(tmp_path: Path) -> tuple[Path, ColumnMappingConfig]:
    """Файл, где у двух борцов идут одинаковые номера эпизодов 1,2,3."""

    path = tmp_path / "two_athletes.xlsx"
    pd.DataFrame(
        {
            "ФИО": [
                "Иванов",
                "Иванов",
                "Иванов",
                "Петров",
                "Петров",
                "Петров",
            ],
            "Эпизод": [1, 2, 3, 1, 2, 3],
            "ЗАП": [
                "ЗАП-Р",
                "ЗАП-Т",
                "ЗАП-Н",
                "удержание",
                "ЗАП-Р",
                "ЗАП-Т",
            ],
        }
    ).to_excel(path, index=False)
    cfg = ColumnMappingConfig(
        sheets={
            "Sheet1": SheetMapping(
                header_rows=[0],
                roles={
                    HiddenGroup.ATHLETE: ["ФИО"],
                    HiddenGroup.EPISODE: ["Эпизод"],
                    HiddenGroup.ZAP: ["ЗАП"],
                },
            )
        }
    )
    return path, cfg


def test_episode_grouping_columns_orders_athlete_bout_episode() -> None:
    sm = SheetMapping(
        header_rows=[0],
        roles={
            HiddenGroup.EPISODE: ["№ эпизода"],
            HiddenGroup.ATHLETE: ["ФИО"],
            HiddenGroup.BOUT: ["Схватка"],
        },
    )
    cols = episode_grouping_columns(sm, ["ФИО", "Схватка", "№ эпизода", "ЗАП"])
    assert cols == ["ФИО", "Схватка", "№ эпизода"]


def test_episode_grouping_columns_skips_missing() -> None:
    sm = SheetMapping(
        header_rows=[0],
        roles={
            HiddenGroup.EPISODE: ["№ эпизода"],
            HiddenGroup.ATHLETE: ["ФИО", "Athlete EN"],
        },
    )
    cols = episode_grouping_columns(sm, ["ФИО", "№ эпизода"])
    assert cols == ["ФИО", "№ эпизода"]


def test_episode_key_skips_empty_pieces() -> None:
    row = pd.Series({"ФИО": "Иванов", "№ эпизода": float("nan")})
    assert episode_key(row, ["ФИО", "№ эпизода"]) == "Иванов"


def test_baseline_episodes_per_sheet_uses_composite_key(tmp_path: Path) -> None:
    path, cfg = _two_athletes_same_numbers(tmp_path)
    frames = load_mapped_sheets(path, cfg)

    baseline, _unknown = build_baseline_with_mapping(
        frames, _empty_audit(), cfg
    )

    # Раньше nunique давал 3 (по «№ эпизода»), теперь должно быть 6 пар.
    assert baseline.episodes_per_sheet["Sheet1"] == 6


def test_hmm_sequences_are_split_per_athlete(tmp_path: Path) -> None:
    path, cfg = _two_athletes_same_numbers(tmp_path)
    frames = load_mapped_sheets(path, cfg)

    sequences, _alphabet = build_observation_sequences(frames, cfg)

    # Шесть уникальных пар (athlete, episode) -> шесть последовательностей.
    assert len(sequences) == 6
    keys = {seq.episode_key for seq in sequences}
    assert "Иванов|1" in keys
    assert "Петров|1" in keys
    assert "Иванов|1" != "Петров|1"
