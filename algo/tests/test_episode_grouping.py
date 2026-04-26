"""Регрессии на ключи группировки эпизодов и серий.

Две разные оси:

* baseline-счёт **эпизодов** — по составному ключу
  ``athlete + bout + episode`` (атомарная единица),
* HMM-**серия** — по ключу ``athlete + bout`` БЕЗ episode, потому что
  один эпизод порождает одно ЗАП-наблюдение и временная динамика
  появляется на уровне последовательности эпизодов одного борца в
  схватке (см. ``DOMAIN_SPEC.md``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.baseline import build_baseline_with_mapping
from hpc_algo.hmm import (
    HMMRunConfig,
    build_observation_sequences,
    evaluate_guards,
)
from hpc_algo.mapping import (
    bout_grouping_columns,
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


def test_bout_grouping_columns_excludes_episode() -> None:
    sm = SheetMapping(
        header_rows=[0],
        roles={
            HiddenGroup.EPISODE: ["№ эпизода"],
            HiddenGroup.ATHLETE: ["ФИО"],
            HiddenGroup.BOUT: ["Схватка"],
        },
    )
    cols = bout_grouping_columns(sm, ["ФИО", "Схватка", "№ эпизода", "ЗАП"])
    assert cols == ["ФИО", "Схватка"]


def test_hmm_sequences_are_grouped_per_athlete(tmp_path: Path) -> None:
    path, cfg = _two_athletes_same_numbers(tmp_path)
    frames = load_mapped_sheets(path, cfg)

    sequences, _alphabet = build_observation_sequences(frames, cfg)

    # Два борца, нет роли bout => две серии длины 3 (по числу строк).
    assert len(sequences) == 2
    by_key = {seq.episode_key: seq for seq in sequences}
    assert "Иванов" in by_key
    assert "Петров" in by_key
    assert len(by_key["Иванов"].tokens) == 3
    assert len(by_key["Петров"].tokens) == 3
    assert by_key["Иванов"].tokens == ["ЗАП-Р", "ЗАП-Т", "ЗАП-Н"]


def test_hmm_short_sequence_guard_blocks_single_step_data(tmp_path: Path) -> None:
    """Если на каждого борца одна строка-эпизод — guard должен сработать."""

    path = tmp_path / "single_step.xlsx"
    pd.DataFrame(
        {
            "ФИО": [f"Боец-{i}" for i in range(40)],
            "Эпизод": list(range(40)),
            "ЗАП": ["ЗАП-Р"] * 40,
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
    frames = load_mapped_sheets(path, cfg)
    baseline, _unknown = build_baseline_with_mapping(frames, _empty_audit(), cfg)
    sequences, alphabet = build_observation_sequences(frames, cfg)

    # У каждого борца ровно одна строка => 40 серий длины 1.
    assert len(sequences) == 40
    assert all(len(s.tokens) == 1 for s in sequences)

    failed = evaluate_guards(baseline, cfg, sequences, alphabet, HMMRunConfig())
    codes = {(w.code, (w.context or {}).get("guard")) for w in failed}
    assert ("hmm.guards_failed", "min_sequence_median") in codes
