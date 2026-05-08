"""Тесты модуля :mod:`hpc_algo.episode_metrics` (TASK_SPEC_013).

Покрытие:

* семантика TS_013: ``action_density`` = mean activations per episode,
  ``activity_evenness`` = нормализованная энтропия per-episode actions;
* ``action_rate_per_second`` — описательная производная;
* ``action_density_first_half`` / ``_second_half`` — половины потока;
* фильтр по ``athlete``;
* классификатор стилей: каждое из правил endurance / speed_power /
  burnout срабатывает на «своих» порогах; иначе ``unclassified`` +
  warning;
* загрузчик YAML: неизвестные стили / правила → warning, не падение.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hpc_algo.episode_metrics import (
    classify_style,
    compute_episode_metrics,
    load_style_thresholds,
)
from hpc_algo.episode_split import RawEpisode
from hpc_algo.schema import EpisodeMetrics, EpisodeRecord, EpisodeState, StyleLabel


def _raw(
    *,
    athlete: str,
    bout_id: str,
    idx: int,
    ep_time: float | None,
    pause_time: float | None,
    features: dict[str, float] | None = None,
    row_index: int = 0,
) -> RawEpisode:
    return RawEpisode(
        row_index=row_index,
        bout_id=bout_id,
        episode_idx_in_bout=idx,
        athlete=athlete,
        episode_num_raw=float(idx),
        episode_time=ep_time,
        pause_time=pause_time,
        score=None,
        feature_values=features or {},
    )


def _rec(*, athlete: str, bout_id: str, idx: int, state: EpisodeState) -> EpisodeRecord:
    return EpisodeRecord(
        athlete=athlete,
        bout_id=bout_id,
        episode_idx=idx,
        state=state,
    )


# ---------------------------------------------------------------------------
# compute_episode_metrics
# ---------------------------------------------------------------------------


def test_empty_inputs_produce_neutral_metrics() -> None:
    m = compute_episode_metrics([], [])
    assert m.episode_count == 0
    assert m.bout_count == 0
    assert m.duration_stats.count == 0
    assert m.duration_stats.p25 is None
    assert m.action_density is None
    assert m.action_rate_per_second is None
    assert m.action_density_first_half is None
    assert m.action_density_second_half is None
    assert m.non_technical_share is None
    assert m.activity_evenness is None
    assert m.style is None


def test_action_density_is_mean_per_episode() -> None:
    raw = [
        _raw(athlete="A", bout_id="b1", idx=1, ep_time=10.0, pause_time=2.0,
             features={"x": 1.0, "y": 0.0}),
        _raw(athlete="A", bout_id="b1", idx=2, ep_time=20.0, pause_time=3.0,
             features={"x": 2.0, "y": 1.0}),
    ]
    records = [
        _rec(athlete="A", bout_id="b1", idx=1, state=EpisodeState.MANOEUVRING),
        _rec(athlete="A", bout_id="b1", idx=2, state=EpisodeState.TECHNICAL_ACTION),
    ]
    m = compute_episode_metrics(raw, records, athlete="A")

    # TS_013: среднее число активаций на эпизод.
    # ep1: 1 + 0 = 1; ep2: 2 + 1 = 3; mean = 2.0.
    assert m.action_density is not None
    assert math.isclose(m.action_density, 2.0)

    # action_rate_per_second — производная для отображения.
    assert m.action_rate_per_second is not None
    assert math.isclose(m.action_rate_per_second, 4.0 / 30.0, rel_tol=1e-6)


def test_duration_stats_include_p25_p75() -> None:
    raw = [
        _raw(athlete="A", bout_id="b1", idx=i + 1, ep_time=float(t), pause_time=None)
        for i, t in enumerate([10.0, 20.0, 30.0, 40.0, 50.0])
    ]
    m = compute_episode_metrics(raw, [], athlete="A")
    d = m.duration_stats
    assert d.count == 5
    assert d.min == 10.0
    assert d.max == 50.0
    # Линейная интерполяция: p25=20, p75=40.
    assert d.p25 is not None and math.isclose(d.p25, 20.0)
    assert d.p75 is not None and math.isclose(d.p75, 40.0)


def test_first_half_and_second_half_density() -> None:
    # 4 эпизода: actions = [3, 3, 1, 0] → first=3, second=0.5.
    raw = [
        _raw(athlete="A", bout_id="b1", idx=1, ep_time=10.0, pause_time=None,
             features={"x": 2.0, "y": 1.0}),  # 3
        _raw(athlete="A", bout_id="b1", idx=2, ep_time=10.0, pause_time=None,
             features={"x": 2.0, "y": 1.0}),  # 3
        _raw(athlete="A", bout_id="b1", idx=3, ep_time=10.0, pause_time=None,
             features={"x": 1.0}),            # 1
        _raw(athlete="A", bout_id="b1", idx=4, ep_time=10.0, pause_time=None,
             features={}),                    # 0
    ]
    m = compute_episode_metrics(raw, [], athlete="A")
    assert m.action_density_first_half is not None
    assert math.isclose(m.action_density_first_half, 3.0)
    assert m.action_density_second_half is not None
    assert math.isclose(m.action_density_second_half, 0.5)


def test_evenness_is_entropy_of_per_episode_actions() -> None:
    # Равномерные действия по эпизодам → 1.0.
    raw_uniform = [
        _raw(athlete="A", bout_id="b1", idx=i + 1, ep_time=10.0, pause_time=None,
             features={"x": 1.0})
        for i in range(5)
    ]
    m_uniform = compute_episode_metrics(raw_uniform, [], athlete="A")
    assert m_uniform.activity_evenness is not None
    assert math.isclose(m_uniform.activity_evenness, 1.0, rel_tol=1e-6)

    # Только один эпизод имеет действия → 0.0 (всё в одной точке).
    raw_single = [
        _raw(athlete="A", bout_id="b1", idx=1, ep_time=10.0, pause_time=None,
             features={"x": 1.0}),
        _raw(athlete="A", bout_id="b1", idx=2, ep_time=10.0, pause_time=None,
             features={}),
        _raw(athlete="A", bout_id="b1", idx=3, ep_time=10.0, pause_time=None,
             features={}),
    ]
    m_single = compute_episode_metrics(raw_single, [], athlete="A")
    assert m_single.activity_evenness == 0.0


def test_non_technical_share_includes_pause() -> None:
    records = [
        _rec(athlete="A", bout_id="b1", idx=1, state=EpisodeState.MANOEUVRING),
        _rec(athlete="A", bout_id="b1", idx=2, state=EpisodeState.PAUSE),
        _rec(athlete="A", bout_id="b1", idx=3, state=EpisodeState.TECHNICAL_ACTION),
    ]
    m = compute_episode_metrics([], records)
    assert m.non_technical_share is not None
    assert math.isclose(m.non_technical_share, 2 / 3, rel_tol=1e-6)


def test_athlete_filter_isolates_subject() -> None:
    raw = [
        _raw(athlete="A", bout_id="bA", idx=1, ep_time=10.0, pause_time=1.0,
             features={"x": 1.0}),
        _raw(athlete="B", bout_id="bB", idx=1, ep_time=20.0, pause_time=2.0,
             features={"x": 1.0}),
    ]
    records = [
        _rec(athlete="A", bout_id="bA", idx=1, state=EpisodeState.MANOEUVRING),
        _rec(athlete="B", bout_id="bB", idx=1, state=EpisodeState.TECHNICAL_ACTION),
    ]
    m_a = compute_episode_metrics(raw, records, athlete="A")
    assert m_a.episode_count == 1
    assert m_a.bout_count == 1
    assert m_a.duration_stats.total == 10.0


def test_invalid_durations_filtered_out() -> None:
    raw = [
        _raw(athlete="A", bout_id="b", idx=1, ep_time=float("nan"), pause_time=None),
        _raw(athlete="A", bout_id="b", idx=2, ep_time=-5.0, pause_time=2.0),
        _raw(athlete="A", bout_id="b", idx=3, ep_time=10.0, pause_time=None),
    ]
    m = compute_episode_metrics(raw, [], athlete="A")
    assert m.duration_stats.count == 1
    assert m.duration_stats.total == 10.0
    assert m.pause_stats.count == 1
    assert m.pause_stats.total == 2.0


# ---------------------------------------------------------------------------
# classify_style
# ---------------------------------------------------------------------------


def _metrics(**overrides) -> EpisodeMetrics:
    base = {
        "episode_count": 0,
        "bout_count": 0,
        "action_density": None,
        "action_rate_per_second": None,
        "action_density_first_half": None,
        "action_density_second_half": None,
        "activity_evenness": None,
        "non_technical_share": None,
    }
    base.update(overrides)
    return EpisodeMetrics(**base)


def test_classify_style_returns_unclassified_without_thresholds() -> None:
    label, w = classify_style(_metrics(episode_count=5), {})
    assert label == StyleLabel.UNCLASSIFIED
    assert w is not None and w.code == "style.no_rule_matched"

    label2, w2 = classify_style(_metrics(episode_count=5), None)
    assert label2 == StyleLabel.UNCLASSIFIED
    assert w2 is not None


def test_classify_style_endurance_rule() -> None:
    thresholds = {
        "endurance": {"min_episode_count": 8, "min_activity_evenness": 0.6},
        "speed_power": {},
        "burnout": {},
    }
    label, w = classify_style(
        _metrics(episode_count=10, activity_evenness=0.7), thresholds
    )
    assert label == StyleLabel.ENDURANCE
    assert w is None


def test_classify_style_speed_power_rule() -> None:
    thresholds = {
        "endurance": {},
        "speed_power": {"max_episode_count": 6, "min_action_density": 1.5},
        "burnout": {},
    }
    label, w = classify_style(
        _metrics(episode_count=4, action_density=2.0), thresholds
    )
    assert label == StyleLabel.SPEED_POWER
    assert w is None


def test_classify_style_burnout_rule() -> None:
    thresholds = {
        "endurance": {},
        "speed_power": {},
        "burnout": {
            "min_action_density_first_half": 1.5,
            "max_action_density_second_half": 0.5,
        },
    }
    label, w = classify_style(
        _metrics(
            episode_count=10,
            action_density_first_half=2.0,
            action_density_second_half=0.2,
        ),
        thresholds,
    )
    assert label == StyleLabel.BURNOUT
    assert w is None


def test_classify_style_no_rule_matched_warns() -> None:
    thresholds = {
        "endurance": {"min_episode_count": 100},
        "speed_power": {"max_episode_count": 1, "min_action_density": 100.0},
        "burnout": {
            "min_action_density_first_half": 100.0,
            "max_action_density_second_half": 0.0,
        },
    }
    label, w = classify_style(
        _metrics(
            episode_count=10,
            action_density=1.0,
            activity_evenness=0.5,
            action_density_first_half=1.0,
            action_density_second_half=1.0,
        ),
        thresholds,
    )
    assert label == StyleLabel.UNCLASSIFIED
    assert w is not None and w.code == "style.no_rule_matched"


def test_classify_style_priority_burnout_before_speed_power() -> None:
    """При одновременном попадании burnout специфичнее → выбирается первым."""

    thresholds = {
        "speed_power": {"max_episode_count": 100, "min_action_density": 0.5},
        "burnout": {
            "min_action_density_first_half": 1.0,
            "max_action_density_second_half": 0.5,
        },
    }
    label, _ = classify_style(
        _metrics(
            episode_count=10,
            action_density=1.0,
            action_density_first_half=2.0,
            action_density_second_half=0.2,
        ),
        thresholds,
    )
    assert label == StyleLabel.BURNOUT


def test_classify_style_missing_metric_does_not_match() -> None:
    """Если метрика None — правило не срабатывает (честнее, чем выдумывать)."""

    thresholds = {
        "endurance": {"min_activity_evenness": 0.1},
    }
    label, w = classify_style(_metrics(episode_count=10), thresholds)
    assert label == StyleLabel.UNCLASSIFIED
    assert w is not None


# ---------------------------------------------------------------------------
# load_style_thresholds
# ---------------------------------------------------------------------------


def test_load_style_thresholds_reads_nested_form(tmp_path: Path) -> None:
    p = tmp_path / "style.yaml"
    p.write_text(
        """
version: 1
thresholds:
  endurance:
    min_episode_count: 8
  speed_power:
    max_episode_count: 6
""",
        encoding="utf-8",
    )
    thresholds, warnings = load_style_thresholds(p)
    assert warnings == []
    assert thresholds["endurance"] == {"min_episode_count": 8.0}
    assert thresholds["speed_power"] == {"max_episode_count": 6.0}


def test_load_style_thresholds_warns_on_unknowns(tmp_path: Path) -> None:
    p = tmp_path / "style.yaml"
    p.write_text(
        """
thresholds:
  hyperdrive:
    min_episode_count: 5
  endurance:
    foo_bar: 10
""",
        encoding="utf-8",
    )
    thresholds, warnings = load_style_thresholds(p)
    codes = {w.code for w in warnings}
    assert "style_thresholds.unknown_style" in codes
    assert "style_thresholds.unknown_rule" in codes
    # Endurance остаётся, но без неизвестного правила.
    assert thresholds["endurance"] == {}


def test_load_style_thresholds_handles_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "style.yaml"
    p.write_text("", encoding="utf-8")
    thresholds, warnings = load_style_thresholds(p)
    assert warnings == []
    # Все известные стили присутствуют, но без правил.
    for k in ("endurance", "speed_power", "burnout"):
        assert thresholds[k] == {}


def test_load_style_thresholds_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "style.yaml"
    p.write_text("- not: a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_style_thresholds(p)
