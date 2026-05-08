"""Тесты разбиения листа `Общее` на bout'ы и эпизоды (TASK_SPEC_011, Фаза A).

Покрываем:

* нормализацию признакового значения (``0/1/2`` ок, ``>2``, текст,
  дробные → ``0`` + warning);
* эвристическое определение служебных колонок после flatten;
* границу bout'а по пустой строке-разделителю;
* границу bout'а по сбросу ``№ эпизода``;
* монотонность ``bout_id``;
* интеграцию с реальным multi-row Excel через :func:`read_episodes_sheet`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.episode_split import (
    BaseColumns,
    _normalize_feature_value,
    detect_base_columns,
    read_episodes_sheet,
    split_into_bouts_and_episodes,
)

# --- normalize_feature_value ---


def test_normalize_zero_one_two_passthrough() -> None:
    for v in (0, 1, 2, 0.0, 1.0, 2.0):
        out, w = _normalize_feature_value(v)
        assert out == float(v)
        assert w is None


def test_normalize_blank_returns_zero() -> None:
    for v in (None, "", "   ", float("nan")):
        out, w = _normalize_feature_value(v)
        assert out == 0.0
        assert w is None


def test_normalize_out_of_range_emits_warning() -> None:
    out, w = _normalize_feature_value(8)
    assert out == 0.0
    assert w is not None
    assert w.code == "data.value_out_of_range"
    assert w.context["value"] == 8.0


def test_normalize_non_numeric_emits_warning() -> None:
    out, w = _normalize_feature_value("abc")
    assert out == 0.0
    assert w is not None
    assert w.code == "data.non_numeric_feature"


def test_normalize_fractional_treated_as_out_of_range() -> None:
    out, w = _normalize_feature_value(1.5)
    assert out == 0.0
    assert w is not None
    assert w.code == "data.value_out_of_range"


# --- detect_base_columns ---


def test_detect_base_columns_finds_athlete_and_episode() -> None:
    cols = [
        "ФИО борца",
        "Технико-тактический эпизод | № эпизода",
        "Технико-тактический эпизод | Время эпизода, с.",
        "Технико-тактический эпизод | Время паузы, с.",
        "Баллы",
        "Стойка | ПС | Вперед",
    ]
    base = detect_base_columns(cols)
    assert base.athlete == "ФИО борца"
    assert base.episode_num == "Технико-тактический эпизод | № эпизода"
    assert base.episode_time == "Технико-тактический эпизод | Время эпизода, с."
    assert base.pause_time == "Технико-тактический эпизод | Время паузы, с."
    assert base.score == "Баллы"


# --- split_into_bouts_and_episodes (synthetic) ---


def _make_synthetic_df() -> tuple[pd.DataFrame, BaseColumns, list[str]]:
    """Минимальный DataFrame, эмулирующий реальный лист после flatten."""

    cols = ["athlete", "ep", "ep_t", "pause_t", "feat_a", "feat_b"]
    rows = [
        # bout 1: ep 1..3
        ["Иванов", 1, 30, 5, 1, 0],
        ["Иванов", 2, 25, 4, 0, 1],
        ["Иванов", 3, 22, None, 1, 0],
        # blank separator
        [None] * 6,
        # bout 2: ep_num resets
        ["Иванов", 1, 18, 6, 0, 1],
        ["Иванов", 2, 20, None, 8, 0],  # value 8 → warning
    ]
    df = pd.DataFrame(rows, columns=cols)
    base = BaseColumns(
        athlete="athlete",
        episode_num="ep",
        episode_time="ep_t",
        pause_time="pause_t",
        score=None,
    )
    return df, base, ["feat_a", "feat_b"]


def test_split_blank_row_closes_bout() -> None:
    df, base, feats = _make_synthetic_df()
    episodes, _ = split_into_bouts_and_episodes(df, base, feats)
    bout_ids = [e.bout_id for e in episodes]
    # 3 in bout_1, 2 in bout_2 (after blank separator), no episode for the blank row itself.
    assert len(episodes) == 5
    assert bout_ids == ["bout_1"] * 3 + ["bout_2"] * 2


def test_split_emits_value_out_of_range_warning() -> None:
    df, base, feats = _make_synthetic_df()
    _, warnings = split_into_bouts_and_episodes(df, base, feats)
    codes = [w.code for w in warnings]
    assert "data.value_out_of_range" in codes


def test_split_episode_num_reset_without_blank_opens_new_bout() -> None:
    """Если между поединками нет пустой строки, но ``№ эпизода`` сбросился к 1, новый bout."""

    cols = ["athlete", "ep", "feat_a"]
    rows = [
        ["Иванов", 1, 1],
        ["Иванов", 2, 0],
        ["Иванов", 1, 1],  # reset
        ["Иванов", 2, 1],
    ]
    df = pd.DataFrame(rows, columns=cols)
    base = BaseColumns(
        athlete="athlete",
        episode_num="ep",
        episode_time=None,
        pause_time=None,
        score=None,
    )
    episodes, _ = split_into_bouts_and_episodes(df, base, ["feat_a"])
    assert [e.bout_id for e in episodes] == ["bout_1", "bout_1", "bout_2", "bout_2"]


def test_split_per_episode_indices_are_one_based_and_monotonic() -> None:
    df, base, feats = _make_synthetic_df()
    episodes, _ = split_into_bouts_and_episodes(df, base, feats)
    by_bout: dict[str, list[int]] = {}
    for e in episodes:
        by_bout.setdefault(e.bout_id, []).append(e.episode_idx_in_bout)
    for indices in by_bout.values():
        assert indices == list(range(1, len(indices) + 1))


# --- end-to-end on real-format multi-row Excel ---


def test_read_and_split_markov_two_bouts(markov_two_bouts_excel: Path) -> None:
    df = read_episodes_sheet(markov_two_bouts_excel, sheet="Общее")
    base = detect_base_columns(df.columns)
    assert base.athlete is not None
    assert base.episode_num is not None
    assert base.pause_time is not None

    feature_cols = [c for c in df.columns if c not in {
        base.athlete, base.episode_num, base.episode_time, base.pause_time, base.score
    }]
    episodes, warnings = split_into_bouts_and_episodes(df, base, feature_cols)

    # Bout 1 = 6 rows, blank separator, Bout 2 = 4 rows. Итого 10 нормализованных эпизодов.
    assert len(episodes) == 10
    bout_ids = sorted({e.bout_id for e in episodes})
    assert bout_ids == ["bout_1", "bout_2"]

    # Защита `>2 → log&skip` сработала на «8» в одной строке.
    codes = {w.code for w in warnings}
    assert "data.value_out_of_range" in codes
