"""Тесты TASK_SPEC_003_1: классификация ЗАП (categorical / binary / count /
empty) и корректный подсчёт событий по каналам.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.baseline import (
    ZAP_KIND_BINARY,
    ZAP_KIND_CATEGORICAL,
    ZAP_KIND_COUNT,
    ZAP_KIND_EMPTY,
    classify_zap_column,
)
from hpc_algo.schema import AnalysisStatus, HiddenGroup

# ---------------------------------------------------------------------------
# classify_zap_column: прямые юниты
# ---------------------------------------------------------------------------


def test_classify_empty() -> None:
    kind, stats = classify_zap_column(pd.Series([None, None, None], dtype="object"))
    assert kind == ZAP_KIND_EMPTY
    assert stats["events"] == 0
    assert stats["total_triggers"] == 0


def test_classify_binary_only_zeros_and_ones() -> None:
    kind, stats = classify_zap_column(pd.Series([0, 1, 0, 1, 1]))
    assert kind == ZAP_KIND_BINARY
    assert stats["events"] == 3
    assert stats["total_triggers"] == 3


def test_classify_count_with_values_above_one() -> None:
    kind, stats = classify_zap_column(pd.Series([0, 0, 2, 1, 3]))
    assert kind == ZAP_KIND_COUNT
    # events: сколько эпизодов с value > 0 (2, 1, 3) = 3
    assert stats["events"] == 3
    assert stats["total_triggers"] == 6


def test_classify_categorical() -> None:
    kind, stats = classify_zap_column(
        pd.Series(["ЗАП-Р", "ЗАП-Т", "удержание", None])
    )
    assert kind == ZAP_KIND_CATEGORICAL
    assert stats["events"] == 3
    assert stats["total_triggers"] == 3
    assert "value_counts" in stats
    assert stats["value_counts"]["ЗАП-Р"] == 1


# ---------------------------------------------------------------------------
# Поведение analyze_source с бинарной/категориальной фикстурой
# ---------------------------------------------------------------------------


def test_categorical_zap_populates_events_by_channel(
    multirow_header_excel: Path,
) -> None:
    cfg = preflight_mapping(multirow_header_excel)
    result = analyze_source(multirow_header_excel, AnalyzeConfig(column_mapping=cfg))

    assert result.status == AnalysisStatus.BASELINE_ONLY
    # Categorical ЗАП-колонка заполняет value_counts.
    assert result.basic_statistics.zap_value_counts
    # И одновременно zap_events_by_channel — хотя бы один канал.
    assert result.basic_statistics.zap_events_by_channel
    # Классификатор отметил колонку как categorical.
    kinds = set(result.basic_statistics.zap_column_kinds.values())
    assert "categorical" in kinds


def test_binary_zap_populates_events_but_not_value_counts(
    multirow_binary_zap_excel: Path,
) -> None:
    cfg = preflight_mapping(multirow_binary_zap_excel)
    # Проверим, что preflight отнёс удержание / болевой к ЗАП.
    roles = cfg.sheets["48"].roles
    assert HiddenGroup.ZAP in roles
    zap_cols = roles[HiddenGroup.ZAP]
    channels = {c.rsplit(" | ", 1)[-1] for c in zap_cols}
    assert {"Удержание", "На руку", "На ногу"}.issubset(channels)

    result = analyze_source(
        multirow_binary_zap_excel, AnalyzeConfig(column_mapping=cfg)
    )
    assert result.status == AnalysisStatus.BASELINE_ONLY

    # Все распознанные ЗАП-колонки классифицированы как binary/count.
    kinds = set(result.basic_statistics.zap_column_kinds.values())
    assert kinds and kinds <= {ZAP_KIND_BINARY, ZAP_KIND_COUNT}

    # События по каналам: удержание 2 эпизода (один с value=1, второй с value=2),
    # на руку — 1, на ногу — 1.
    channels_map = result.basic_statistics.zap_events_by_channel
    assert channels_map["Удержание"] == 2
    assert channels_map["На руку"] == 1
    assert channels_map["На ногу"] == 1

    # total_triggers для «Удержание» должен быть 3 (1 + 2).
    udrzh_path = next(
        k for k in result.basic_statistics.zap_total_triggers
        if k.endswith("Удержание")
    )
    assert result.basic_statistics.zap_total_triggers[udrzh_path] == 3

    # value_counts для бинарных ЗАП-колонок НЕ заполняются
    # (чтобы UI не показывал 0/1-гистограмму).
    assert result.basic_statistics.zap_value_counts == {}


def test_binary_zap_yields_chart_zap_events_by_channel(
    multirow_binary_zap_excel: Path,
) -> None:
    cfg = preflight_mapping(multirow_binary_zap_excel)
    result = analyze_source(
        multirow_binary_zap_excel, AnalyzeConfig(column_mapping=cfg)
    )
    chart_ids = [c.id for c in result.charts]
    assert "zap_events_by_channel" in chart_ids
    # Так как ЗАП тут только binary/count, категорийного графика быть
    # не должно (payload hidden_group_value_counts[ЗАП] состоит из
    # events/triggers, а не меток).
    assert "group_ЗАП" not in chart_ids


def test_report_text_mentions_zap_channels(
    multirow_binary_zap_excel: Path,
) -> None:
    cfg = preflight_mapping(multirow_binary_zap_excel)
    result = analyze_source(
        multirow_binary_zap_excel, AnalyzeConfig(column_mapping=cfg)
    )
    assert "ЗАП-события по каналам" in result.report
    assert "Удержание" in result.report
