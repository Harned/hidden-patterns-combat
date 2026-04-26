"""Baseline-распределения и chart-ready данные.

Две ветки:

* ``build_baseline`` (без mapping) — безопасный fallback: распределения
  ЗАП только для колонок, прошедших эвристический порог детекции;
  missing-values график;
* ``build_baseline_with_mapping`` — после применения подтверждённого
  :class:`ColumnMappingConfig`: распределения по всем четырём доменным
  группам, time-статистики, числа эпизодов на лист. ЗАП-колонки
  классифицируются на ``categorical``/``binary``/``count``/``empty``
  (см. :func:`classify_zap_column`), что отражается в
  ``zap_events_by_channel`` (TASK_SPEC_003_1).

В любой из веток модуль **не** рассчитывает скрытые состояния, Viterbi
или вероятностные профили — это зона ответственности будущей HMM-ветки.
"""

from __future__ import annotations

import math

import pandas as pd

from hpc_algo.detection import strong_zap_candidates
from hpc_algo.loading import LoadedExcel
from hpc_algo.mapping import episode_grouping_columns, episode_key
from hpc_algo.schema import (
    AuditReport,
    BaselineReport,
    ChartData,
    ColumnDetectionReport,
    ColumnMappingConfig,
    HiddenGroup,
    TimeStats,
)

# Разделитель flatten-имени колонки. Используется для выделения атомарного
# (последнего) уровня — «канала» ЗАП-наблюдения.
_FLATTEN_SEP = " | "


ZAP_KIND_CATEGORICAL = "categorical"
ZAP_KIND_BINARY = "binary"
ZAP_KIND_COUNT = "count"
ZAP_KIND_EMPTY = "empty"

# ---------------------------------------------------------------------------
# Без mapping: прежний консервативный baseline
# ---------------------------------------------------------------------------


def build_baseline(
    loaded: LoadedExcel,
    audit: AuditReport,
    detection: ColumnDetectionReport,
) -> BaselineReport:
    """Собрать :class:`BaselineReport` без подтверждённого mapping."""

    zap_counts: dict[str, dict[str, int]] = {}
    notes: list[str] = []

    confident_zap = strong_zap_candidates(detection)
    for cand in confident_zap:
        df = loaded.sheets.get(cand.sheet)
        if df is None or cand.column not in df.columns:
            continue
        series = df[cand.column].dropna()
        if series.empty:
            continue
        counts = series.astype(str).value_counts().sort_values(ascending=False)
        zap_counts[f"{cand.sheet}.{cand.column}"] = {
            str(k): int(v) for k, v in counts.items()
        }

    if not confident_zap:
        notes.append(
            "Ни одна ЗАП-колонка не прошла порог уверенной детекции. "
            "Распределения ЗАП не рассчитываются."
        )
    else:
        notes.append(
            "Распределения ЗАП рассчитаны только для колонок, прошедших порог "
            "уверенной детекции; это baseline-статистика, а не диагностика."
        )

    missing_per_column: dict[str, dict[str, int]] = {}
    for sheet in audit.sheets:
        if not sheet.columns:
            continue
        missing_per_column[sheet.name] = {
            col.name: col.null_count for col in sheet.columns
        }

    return BaselineReport(
        zap_value_counts=zap_counts,
        missing_values_per_column=missing_per_column,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# С применённым mapping: распределения по всем 4 группам + time-stats
# ---------------------------------------------------------------------------


def _safe_value_counts(series: pd.Series) -> dict[str, int]:
    """Value counts только по непустым значениям, как строки."""

    dropped = series.dropna()
    if dropped.empty:
        return {}
    counts = dropped.astype(str).value_counts().sort_values(ascending=False)
    return {str(k): int(v) for k, v in counts.items()}


def classify_zap_column(
    series: pd.Series,
) -> tuple[str, dict[str, int | dict[str, int]]]:
    """Определить тип ЗАП-колонки и посчитать честные метрики событий.

    Возвращает кортеж ``(kind, stats)``:

    * ``kind`` ∈ {``categorical``, ``binary``, ``count``, ``empty``};
    * ``stats`` — словарь с полями:
        - ``events`` — число эпизодов, где произошло событие
          (для категориальной — число непустых значений);
        - ``total_triggers`` — сумма целочисленных значений (для
          binary/count) или то же, что ``events`` (для categorical);
        - ``value_counts`` (только для categorical) — распределение меток.
    """

    non_null = series.dropna()
    if non_null.empty:
        return ZAP_KIND_EMPTY, {"events": 0, "total_triggers": 0}

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().sum() == len(non_null):
        # Это числовая колонка.
        unique = set(numeric.unique().tolist())
        events = int((numeric > 0).sum())
        total = int(numeric.clip(lower=0).sum())
        if unique <= {0.0, 1.0}:
            return ZAP_KIND_BINARY, {"events": events, "total_triggers": total}
        return ZAP_KIND_COUNT, {"events": events, "total_triggers": total}

    counts = non_null.astype(str).value_counts().sort_values(ascending=False)
    value_counts = {str(k): int(v) for k, v in counts.items()}
    events = int(len(non_null))
    return (
        ZAP_KIND_CATEGORICAL,
        {
            "events": events,
            "total_triggers": events,
            "value_counts": value_counts,
        },
    )


def _channel_from_flat_name(flat_name: str) -> str:
    """Получить `channel_label` — атомарный (последний) уровень flatten-имени."""

    parts = [p.strip() for p in flat_name.split(_FLATTEN_SEP) if p.strip()]
    return parts[-1] if parts else flat_name


def _time_stats(series: pd.Series) -> TimeStats:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    null_count = int(series.isna().sum())
    if non_null.empty:
        return TimeStats(count=0, null_count=null_count)
    return TimeStats(
        count=int(non_null.count()),
        min=float(non_null.min()),
        max=float(non_null.max()),
        mean=float(round(non_null.mean(), 4)),
        median=float(round(non_null.median(), 4)),
        null_count=null_count,
    )


def build_baseline_with_mapping(
    frames: dict[str, pd.DataFrame],
    audit: AuditReport,
    config: ColumnMappingConfig,
) -> tuple[BaselineReport, list[str]]:
    """Собрать baseline по подтверждённому mapping.

    Возвращает ``(report, unknown_columns)`` — список пар ``sheet.column``,
    на которые mapping ссылается, но которых нет после flatten (они
    превратятся в warnings).
    """

    hidden_counts: dict[str, dict[str, dict[str, int]]] = {}
    hidden_totals: dict[str, int] = {}
    time_stats: dict[str, TimeStats] = {}
    episodes_per_sheet: dict[str, int] = {}
    empty_data_rows_per_sheet: dict[str, int] = {}
    zap_counts: dict[str, dict[str, int]] = {}
    zap_column_kinds: dict[str, str] = {}
    zap_events: dict[str, int] = {}
    zap_total_triggers: dict[str, int] = {}
    zap_events_by_channel: dict[str, int] = {}
    unknown: list[str] = []

    for sheet_name, sheet_mapping in config.sheets.items():
        df = frames.get(sheet_name)
        if df is None:
            continue

        # Считаем число полностью пустых data-строк (после применения header_rows
        # и data_start_row, что уже сделал mapping_mod.load_mapped_sheets).
        if not df.empty:
            empty_data_rows_per_sheet[sheet_name] = int(
                df.isna().all(axis=1).sum()
            )
        else:
            empty_data_rows_per_sheet[sheet_name] = 0

        if HiddenGroup.EPISODE in sheet_mapping.roles:
            for col in sheet_mapping.roles[HiddenGroup.EPISODE]:
                if col not in df.columns:
                    unknown.append(f"{sheet_name}.{col}")

            # Эпизод уникален в рамках листа, борца и схватки. Голый
            # "№ эпизода" обычно перезапускается у каждого борца, поэтому
            # nunique по одной колонке систематически занижает счёт.
            grouping_cols = episode_grouping_columns(
                sheet_mapping, df.columns
            )
            if grouping_cols and not df.empty:
                keys: set[str] = set()
                for _, row in df.iterrows():
                    key = episode_key(row, grouping_cols)
                    if key:
                        keys.add(key)
                episodes_per_sheet[sheet_name] = len(keys)
            else:
                episodes_per_sheet[sheet_name] = 0

        for role, cols in sheet_mapping.roles.items():
            if role == HiddenGroup.TIME:
                for col in cols:
                    if col not in df.columns:
                        unknown.append(f"{sheet_name}.{col}")
                        continue
                    time_stats[f"{sheet_name}.{col}"] = _time_stats(df[col])
                continue

            if role in {HiddenGroup.ATHLETE, HiddenGroup.BOUT, HiddenGroup.WEIGHT}:
                # Не собираем распределения — эти поля служебные.
                continue

            if role == HiddenGroup.ZAP:
                for col in cols:
                    if col not in df.columns:
                        unknown.append(f"{sheet_name}.{col}")
                        continue
                    kind, stats = classify_zap_column(df[col])
                    path = f"{sheet_name}.{col}"
                    zap_column_kinds[path] = kind

                    events = int(stats.get("events", 0) or 0)
                    triggers = int(stats.get("total_triggers", 0) or 0)
                    zap_events[path] = events
                    zap_total_triggers[path] = triggers

                    if events == 0:
                        continue

                    channel = _channel_from_flat_name(col)
                    zap_events_by_channel[channel] = (
                        zap_events_by_channel.get(channel, 0) + events
                    )

                    bucket = hidden_counts.setdefault(role.value, {})
                    if kind == ZAP_KIND_CATEGORICAL:
                        value_counts = stats.get("value_counts") or {}
                        if isinstance(value_counts, dict) and value_counts:
                            bucket[path] = value_counts
                            zap_counts[path] = value_counts
                    else:
                        # Для binary/count сохраняем сводку в "events/triggers"
                        # формате — так UI видит channel → сколько срабатываний.
                        bucket[path] = {
                            "events": events,
                            "triggers": triggers,
                        }

                    hidden_totals[role.value] = hidden_totals.get(role.value, 0) + events
                continue

            for col in cols:
                if col not in df.columns:
                    unknown.append(f"{sheet_name}.{col}")
                    continue
                counts = _safe_value_counts(df[col])
                if not counts:
                    continue

                path = f"{sheet_name}.{col}"
                bucket = hidden_counts.setdefault(role.value, {})
                bucket[path] = counts
                hidden_totals[role.value] = hidden_totals.get(
                    role.value, 0
                ) + sum(counts.values())

    notes: list[str] = []
    for group in (
        HiddenGroup.ZAP,
        HiddenGroup.MANEUVERING,
        HiddenGroup.KFV,
        HiddenGroup.VUP,
    ):
        total = hidden_totals.get(group.value, 0)
        if total:
            notes.append(
                f"Группа {group.value}: собрано {total} наблюдений по "
                f"{len(hidden_counts.get(group.value, {}))} колонкам."
            )

    if unknown:
        notes.append(
            f"Колонки из mapping, не найденные на листе: {len(unknown)}. "
            "Они проигнорированы и отражены в warnings."
        )

    # Графики пропусков строим только по листам из mapping: чтобы избежать
    # путаницы (графики «Unnamed: N» для листов, которые пользователь сам
    # исключил из обработки) и согласовать набор листов с тем, что реально
    # пошло в анализ.
    mapped_sheets = set(config.sheets.keys())
    missing_per_column: dict[str, dict[str, int]] = {}
    for sheet in audit.sheets:
        if sheet.name not in mapped_sheets:
            continue
        if sheet.columns:
            missing_per_column[sheet.name] = {
                col.name: col.null_count for col in sheet.columns
            }

    # Специальное примечание про ЗАП-режим.
    binary_or_count = [
        path for path, kind in zap_column_kinds.items()
        if kind in {ZAP_KIND_BINARY, ZAP_KIND_COUNT}
    ]
    categorical = [
        path for path, kind in zap_column_kinds.items()
        if kind == ZAP_KIND_CATEGORICAL
    ]
    if binary_or_count:
        notes.append(
            f"ЗАП закодирован бинарно/счётчиком в {len(binary_or_count)} колонках. "
            "Для них агрегируется число событий по каналам (zap_events_by_channel)."
        )
    if categorical:
        notes.append(
            f"ЗАП как категориальные метки — в {len(categorical)} колонках. "
            "Для них сохранены стандартные распределения value_counts."
        )

    report = BaselineReport(
        zap_value_counts=zap_counts,
        missing_values_per_column=missing_per_column,
        hidden_group_value_counts=hidden_counts,
        hidden_group_totals=hidden_totals,
        zap_column_kinds=zap_column_kinds,
        zap_events=zap_events,
        zap_total_triggers=zap_total_triggers,
        zap_events_by_channel=zap_events_by_channel,
        time_statistics=time_stats,
        episodes_per_sheet=episodes_per_sheet,
        empty_data_rows_per_sheet=empty_data_rows_per_sheet,
        notes=notes,
    )
    return report, unknown


# ---------------------------------------------------------------------------
# Chart-ready JSON
# ---------------------------------------------------------------------------


def _merge_counts(values: dict[str, dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for counts in values.values():
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    return total


def build_charts(baseline: BaselineReport) -> list[ChartData]:
    """Построить chart-ready JSON на основе baseline.

    Для MVP графики простые: bar/hbar. Heatmap будущей HMM — отдельный этап.
    """

    charts: list[ChartData] = []

    # 1a. ЗАП-события по каналам (binary/count-колонки).
    if baseline.zap_events_by_channel:
        items = sorted(
            baseline.zap_events_by_channel.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        head, tail = items[:20], items[20:]
        x = [k for k, _ in head]
        y = [v for _, v in head]
        if tail:
            x.append("прочее")
            y.append(sum(v for _, v in tail))
        charts.append(
            ChartData(
                id="zap_events_by_channel",
                title="ЗАП-события по каналам",
                kind="bar",
                x=x,
                y=y,
                meta={
                    "group": HiddenGroup.ZAP.value,
                    "channels": len(baseline.zap_events_by_channel),
                    "shown": len(head),
                    "tail_bucketed": bool(tail),
                    "semantics": "events",
                },
            )
        )

    # 1b. Распределения по каждой группе — для категориальных ЗАП и
    # остальных предметных групп. ЗАП-группа рисуется только если в
    # hidden_group_value_counts есть category-like payload (только для
    # categorical ЗАП; binary/count в hidden_counts хранятся как
    # {"events","triggers"} и в bar-график не идут).
    group_titles = {
        HiddenGroup.ZAP.value: "Распределение ЗАП (категории)",
        HiddenGroup.MANEUVERING.value: "Маневрирование: распределение",
        HiddenGroup.KFV.value: "КФВ: распределение",
        HiddenGroup.VUP.value: "ВУП: распределение",
    }
    for group_value, title in group_titles.items():
        bucket = baseline.hidden_group_value_counts.get(group_value)
        if not bucket:
            continue

        if group_value == HiddenGroup.ZAP.value:
            # Фильтруем только категориальные payload'ы
            bucket = {
                path: counts
                for path, counts in bucket.items()
                if "events" not in counts or len(counts) > 2
            }
            if not bucket:
                continue

        merged = _merge_counts(bucket)
        if not merged:
            continue
        items = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        head, tail = items[:20], items[20:]
        x = [k for k, _ in head]
        y = [v for _, v in head]
        if tail:
            x.append("прочее")
            y.append(sum(v for _, v in tail))
        charts.append(
            ChartData(
                id=f"group_{group_value}",
                title=title,
                kind="bar",
                x=x,
                y=y,
                meta={
                    "group": group_value,
                    "columns": len(bucket),
                    "shown": len(head),
                    "tail_bucketed": bool(tail),
                },
            )
        )

    # 2. Fallback: если mapping не применялся, используем zap_value_counts per-column.
    if not baseline.hidden_group_value_counts and baseline.zap_value_counts:
        for idx, (path, counts) in enumerate(sorted(baseline.zap_value_counts.items())):
            items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            head, tail = items[:20], items[20:]
            x = [k for k, _ in head]
            y = [v for _, v in head]
            if tail:
                x.append("прочее")
                y.append(sum(v for _, v in tail))
            charts.append(
                ChartData(
                    id=f"zap_distribution_{idx}",
                    title=f"Распределение ЗАП: {path}",
                    kind="bar",
                    x=x,
                    y=y,
                    meta={"source_column": path, "shown": len(head)},
                )
            )

    # 2.5. Полностью пустые data-строки по листам (mapping-ветка).
    if baseline.empty_data_rows_per_sheet:
        items = sorted(
            baseline.empty_data_rows_per_sheet.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        charts.append(
            ChartData(
                id="empty_data_rows_per_sheet",
                title="Полностью пустые строки данных по листам",
                kind="hbar",
                x=[v for _, v in items],
                y=[k for k, _ in items],
                meta={
                    "shown": len(items),
                    "semantics": "empty_rows",
                },
            )
        )

    # 3. Missing-values per sheet.
    for sheet_name, cols in baseline.missing_values_per_column.items():
        items = [(k, v) for k, v in cols.items() if v > 0]
        items.sort(key=lambda kv: kv[1], reverse=True)
        if not items:
            continue
        charts.append(
            ChartData(
                id=f"missing_values__{sheet_name}",
                title=f"Пропуски по колонкам: {sheet_name}",
                kind="hbar",
                x=[v for _, v in items[:30]],
                y=[k for k, _ in items[:30]],
                meta={
                    "sheet": sheet_name,
                    "shown": min(30, len(items)),
                    "total": len(items),
                },
            )
        )

    # 4. Time-статистики (для одной сводной hbar).
    if baseline.time_statistics:
        sheets = list(baseline.time_statistics.keys())
        means = [
            float(stats.mean) if stats.mean is not None and not math.isnan(stats.mean) else 0.0
            for stats in baseline.time_statistics.values()
        ]
        charts.append(
            ChartData(
                id="time_stats_mean",
                title="Время эпизода: среднее по колонкам",
                kind="hbar",
                x=means,
                y=sheets,
                meta={"shown": len(sheets)},
            )
        )

    return charts


__all__ = [
    "build_baseline",
    "build_baseline_with_mapping",
    "build_charts",
]
