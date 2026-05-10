"""Публичный интерфейс processing module.

Главная точка входа — :func:`analyze_source`. Возвращает
:class:`AnalysisResult` и никогда не бросает исключений наверх: любая
проблема оказывается в поле ``errors`` результата.

Также доступен :func:`preflight_mapping` — safe-to-call функция,
возвращающая предлагаемый :class:`ColumnMappingConfig` для данного файла.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from hpc_algo import hmm as hmm_mod
from hpc_algo import mapping as mapping_mod
from hpc_algo.audit import (
    build_audit,
    detect_totals_row_indices,
    drop_totals_rows,
    filter_audit_to_sheets,
)
from hpc_algo.baseline import (
    build_baseline,
    build_baseline_with_mapping,
    build_charts,
)
from hpc_algo.detection import detect_columns, strong_zap_candidates, weak_candidates
from hpc_algo.hmm import HMMRunConfig
from hpc_algo.loading import ExcelLoadError, LoadedExcel, load_excel
from hpc_algo.schema import (
    AnalysisResult,
    AnalysisStatus,
    AuditReport,
    BaselineReport,
    ChartData,
    ColumnDetectionReport,
    ColumnMappingConfig,
    HiddenGroup,
    HMMResult,
    SheetMapping,
    SourceMetadata,
    WarningItem,
    WarningSeverity,
)
from hpc_algo.trainer import build_athlete_episode_rollup

# Имя синтетической колонки, которая хранит виртуальную схватку,
# восстановленную по сбросам нумерации эпизодов (см.
# :func:`hpc_algo.mapping.virtual_bout_series`). Имя начинается с
# подчёркивания, чтобы исключить случайное совпадение с реальной
# колонкой пользователя.
_VIRTUAL_BOUT_COLUMN = "_virtual_bout_"


@dataclass
class AnalyzeConfig:
    """Конфигурация анализа."""

    max_preview_rows: int = 5
    column_mapping: ColumnMappingConfig | None = None

    # HMM-ветка (TASK_SPEC_004 / TASK_SPEC_005 / TASK_SPEC_008).
    enable_hmm: bool = True
    hmm_mode: str = "auto"  # auto | detailed | basic | off
    observation_emission: str = "categorical"  # categorical | bernoulli
    hmm_seed: int = 42
    hmm_min_episodes: int = 30
    hmm_min_zap_events: int = 20
    hmm_min_episodes_detailed: int = 120
    hmm_min_zap_events_detailed: int = 60
    hmm_n_iter: int = 50

    extra: dict[str, Any] = field(default_factory=dict)

    def hmm_run_config(self) -> HMMRunConfig:
        return HMMRunConfig(
            enable_hmm=self.enable_hmm,
            mode=self.hmm_mode if self.enable_hmm else "off",
            observation_emission=self.observation_emission,
            min_episodes=self.hmm_min_episodes,
            min_zap_events=self.hmm_min_zap_events,
            min_episodes_detailed=self.hmm_min_episodes_detailed,
            min_zap_events_detailed=self.hmm_min_zap_events_detailed,
            n_iter=self.hmm_n_iter,
            random_seed=self.hmm_seed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_audit() -> AuditReport:
    return AuditReport(sheets=[], total_rows=0, total_cells=0, overall_null_ratio=0.0)


def _failed_result(
    source_path: Path,
    size_bytes: int,
    error: WarningItem,
) -> AnalysisResult:
    return AnalysisResult(
        status=AnalysisStatus.FAILED,
        source_metadata=SourceMetadata(
            filename=source_path.name,
            size_bytes=size_bytes,
        ),
        data_audit=_empty_audit(),
        detected_columns=ColumnDetectionReport(),
        basic_statistics=BaselineReport(),
        errors=[error],
        report=f"Анализ не выполнен: {error.message}",
    )


def _build_multirow_warning(audit: AuditReport) -> WarningItem | None:
    sheets_with_multirow: list[str] = []
    for sheet in audit.sheets:
        total = len(sheet.columns)
        if not total:
            continue
        unnamed = sum(1 for c in sheet.columns if c.name.startswith("Unnamed:"))
        if unnamed / total >= 0.5:
            sheets_with_multirow.append(sheet.name)
    if not sheets_with_multirow:
        return None
    return WarningItem(
        code="audit.possible_multirow_header",
        message=(
            "Похоже на многострочный заголовок: на листах "
            f"{', '.join(sheets_with_multirow)} большинство колонок "
            "называются 'Unnamed: N'. Используйте column_mapping с "
            "header_rows=[0,1,2] (или подходящим диапазоном)."
        ),
        severity=WarningSeverity.WARNING,
        context={"sheets": sheets_with_multirow},
    )


# ---------------------------------------------------------------------------
# Heuristic (no mapping) path
# ---------------------------------------------------------------------------


def _decide_status_heuristic(
    detection: ColumnDetectionReport,
    audit: AuditReport,
) -> AnalysisStatus:
    if audit.total_rows == 0:
        return AnalysisStatus.AUDIT_ONLY

    has_strong_zap = any(
        c.group == HiddenGroup.ZAP and c.score >= 0.7 for c in detection.candidates
    )

    required = {
        HiddenGroup.ZAP,
        HiddenGroup.MANEUVERING,
        HiddenGroup.KFV,
        HiddenGroup.VUP,
        HiddenGroup.TIME,
    }
    if required.issubset(set(detection.detected_groups)):
        return AnalysisStatus.BASELINE_ONLY
    if has_strong_zap:
        return AnalysisStatus.BASELINE_ONLY
    return AnalysisStatus.NEEDS_COLUMN_MAPPING


def _build_heuristic_warnings(
    detection: ColumnDetectionReport,
    audit: AuditReport,
    status: AnalysisStatus,
) -> list[WarningItem]:
    warnings: list[WarningItem] = []

    if audit.total_rows == 0:
        warnings.append(
            WarningItem(
                code="audit.empty_file",
                message="Файл не содержит данных ни на одном листе.",
                severity=WarningSeverity.ERROR,
            )
        )

    if audit.overall_null_ratio >= 0.5:
        warnings.append(
            WarningItem(
                code="audit.many_missing",
                message=(
                    f"Доля пропусков по всему файлу высокая: "
                    f"{audit.overall_null_ratio * 100:.1f}%."
                ),
                severity=WarningSeverity.WARNING,
                context={"overall_null_ratio": audit.overall_null_ratio},
            )
        )

    mr = _build_multirow_warning(audit)
    if mr is not None:
        warnings.append(mr)

    for sheet in audit.sheets:
        if sheet.n_rows == 0:
            warnings.append(
                WarningItem(
                    code="audit.empty_sheet",
                    message=f"Лист '{sheet.name}' пустой.",
                    severity=WarningSeverity.INFO,
                    context={"sheet": sheet.name},
                )
            )
        for note in sheet.suspicious:
            warnings.append(
                WarningItem(
                    code="audit.suspicious",
                    message=f"[{sheet.name}] {note}",
                    severity=WarningSeverity.INFO,
                    context={"sheet": sheet.name},
                )
            )

    if not strong_zap_candidates(detection):
        warnings.append(
            WarningItem(
                code="detection.no_strong_zap",
                message=(
                    "Не найдено колонок, уверенно соответствующих ЗАП (observations). "
                    "Распределения ЗАП и диагностика в этом запуске не рассчитываются."
                ),
                severity=WarningSeverity.WARNING,
            )
        )

    if weak := weak_candidates(detection):
        warnings.append(
            WarningItem(
                code="detection.weak_candidates",
                message=(
                    "Найдены слабые кандидаты колонок. Они требуют ручного "
                    "подтверждения через column_mapping."
                ),
                severity=WarningSeverity.INFO,
                context={
                    "count": len(weak),
                    "candidates": [
                        {
                            "group": c.group.value,
                            "sheet": c.sheet,
                            "column": c.column,
                            "score": c.score,
                        }
                        for c in weak[:20]
                    ],
                },
            )
        )

    if detection.missing_groups:
        warnings.append(
            WarningItem(
                code="detection.missing_required_groups",
                message=(
                    "Не удалось уверенно распознать обязательные группы: "
                    + ", ".join(g.value for g in detection.missing_groups)
                    + ". Полноценная HMM в этом запуске невозможна."
                ),
                severity=WarningSeverity.WARNING,
                context={"missing": [g.value for g in detection.missing_groups]},
            )
        )

    if status == AnalysisStatus.NEEDS_COLUMN_MAPPING:
        warnings.append(
            WarningItem(
                code="status.needs_column_mapping",
                message=(
                    "Структура Excel не позволяет надёжно определить группы скрытых "
                    "состояний и ЗАП. Требуется ручное сопоставление колонок."
                ),
                severity=WarningSeverity.WARNING,
            )
        )

    return warnings


def _build_report(
    status: AnalysisStatus,
    source: SourceMetadata,
    audit: AuditReport,
    detection: ColumnDetectionReport,
    baseline: BaselineReport,
    mapping_applied: bool,
    hmm: HMMResult | None = None,
) -> str:
    status_messages = {
        AnalysisStatus.AUDIT_ONLY: (
            "Статус: audit_only — удалось выполнить только аудит файла."
        ),
        AnalysisStatus.NEEDS_COLUMN_MAPPING: (
            "Статус: needs_column_mapping — требуется ручное сопоставление колонок."
        ),
        AnalysisStatus.BASELINE_ONLY: (
            "Статус: baseline_only — рассчитаны только базовые описательные "
            "характеристики. Диагностика скрытой траектории не выполнена."
        ),
        AnalysisStatus.HMM_READY: (
            "Статус: hmm_ready — обучена HMM (см. applied variant), прошли "
            "guard-ы по числу эпизодов/событий, sanity-check матрицы переходов "
            "и BIC-гейт. Все выводы остаются вероятностными."
        ),
        AnalysisStatus.HMM_LOW_SIGNAL: (
            "Статус: hmm_low_signal — HMM обучена, но ЗАП-сигнал в данных "
            "разрежённый: эмиссии состояний близки, и часть Viterbi-пути "
            "восстанавливается приором, а не наблюдениями. Интерпретация "
            "модели требует осторожности."
        ),
        AnalysisStatus.FAILED: "Статус: failed — см. errors.",
    }

    lines: list[str] = []

    if mapping_applied:
        # Honesty layer: показываем не «Наблюдения по группам», а
        # «Непустых ячеек по ролям» — чтобы пользователь не путал
        # количество заполненных клеток в таблице с количеством
        # эпизодов или ЗАП-событий.
        totals = baseline.hidden_group_totals
        n_role_columns = sum(
            len(cols) for cols in (baseline.hidden_group_value_counts or {}).values()
        )
        if totals:
            suffix = (
                f" (по {n_role_columns} колонкам)" if n_role_columns else ""
            )
            lines.append(
                f"Непустых ячеек по ролям{suffix}: "
                + ", ".join(f"{k}={v}" for k, v in totals.items())
                + "."
            )
        # Эпизоды и доля без ZAP — главный сигнал для тренера/аналитика
        # о плотности наблюдений.
        total_episodes = sum(baseline.episodes_per_sheet.values())
        no_zap = hmm.no_zap_trajectories if hmm is not None else None
        if total_episodes:
            if no_zap is not None:
                pct = (no_zap / total_episodes * 100.0) if total_episodes else 0.0
                lines.append(
                    f"Эпизодов без ЗАП-маркера: {no_zap} из {total_episodes} "
                    f"({pct:.1f}%) — серии без ЗАП исключаются из обучения HMM, "
                    "Viterbi для них восстанавливается по приору."
                )
            else:
                lines.append(f"Эпизодов всего (по mapping): {total_episodes}.")
        if baseline.zap_events_by_channel:
            top = sorted(
                baseline.zap_events_by_channel.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]
            lines.append(
                "ЗАП-события по каналам (топ-5): "
                + ", ".join(f"{k}={v}" for k, v in top)
                + "."
            )
        if baseline.time_statistics:
            lines.append(
                f"Рассчитаны time-статистики для {len(baseline.time_statistics)} колонок."
            )
        if hmm is not None:
            applied = next(
                (a for a in hmm.tried_variants if a.status == "applied"),
                None,
            )
            applied_name = (
                applied.variant if applied is not None else hmm.parameters.variant
            )
            other = [a for a in hmm.tried_variants if a.status != "applied"]
            if other:
                rejected_parts = [
                    f"{a.variant}: {a.status}"
                    + (f" ({a.reason})" if a.reason else "")
                    for a in other
                ]
                lines.append(
                    f"Применён вариант HMM: {applied_name}. "
                    + "Остальные попытки — "
                    + "; ".join(rejected_parts)
                    + "."
                )
            else:
                lines.append(f"Применён вариант HMM: {applied_name}.")
            if hmm.average_confidence is not None:
                lines.append(
                    "Средняя уверенность Viterbi-пути: "
                    f"{hmm.average_confidence * 100:.1f}% "
                    "(чем ниже — тем больше эпизодов следуют приору)."
                )
    else:
        if detection.detected_groups:
            lines.append(
                "Уверенно распознанные группы: "
                + ", ".join(g.value for g in detection.detected_groups)
                + "."
            )
        else:
            lines.append("Ни одна предметная группа не распознана уверенно.")
        if detection.missing_groups:
            lines.append(
                "Не распознаны: "
                + ", ".join(g.value for g in detection.missing_groups)
                + "."
            )

    lines.append("")
    lines.append(status_messages.get(status, f"Статус: {status.value}."))

    lines.append("")
    lines.append("Сводка по данным (аудит):")
    lines.append(f"Файл: {source.filename}")
    lines.append(f"Листов: {source.sheet_count}. Всего строк: {audit.total_rows}.")
    lines.append(f"Доля пропусков по файлу: {audit.overall_null_ratio * 100:.1f}%.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mapping-driven path
# ---------------------------------------------------------------------------


def _has_zap_values(baseline: BaselineReport) -> bool:
    # Категориальные ЗАП-метки.
    zap_bucket = baseline.hidden_group_value_counts.get(HiddenGroup.ZAP.value, {})
    if any(bool(v) for v in zap_bucket.values()):
        return True
    # Binary/count-события по каналам.
    return any(v > 0 for v in baseline.zap_events_by_channel.values())


def _decide_status_with_mapping(baseline: BaselineReport) -> AnalysisStatus:
    if _has_zap_values(baseline):
        return AnalysisStatus.BASELINE_ONLY
    return AnalysisStatus.NEEDS_COLUMN_MAPPING


def _mapping_warnings(
    config: ColumnMappingConfig,
    baseline: BaselineReport,
    unknown_columns: list[str],
    status: AnalysisStatus,
) -> list[WarningItem]:
    warnings: list[WarningItem] = []

    if unknown_columns:
        warnings.append(
            WarningItem(
                code="mapping.unknown_column",
                message=(
                    "Mapping ссылается на колонки, которых не оказалось на листах: "
                    f"{len(unknown_columns)}. Проверьте header_rows и имена."
                ),
                severity=WarningSeverity.WARNING,
                context={"columns": unknown_columns[:50]},
            )
        )

    missing_groups: list[str] = []
    required_for_hmm_ready = (
        HiddenGroup.ZAP,
        HiddenGroup.MANEUVERING,
        HiddenGroup.KFV,
        HiddenGroup.VUP,
    )
    for group in required_for_hmm_ready:
        if baseline.hidden_group_totals.get(group.value, 0) == 0:
            missing_groups.append(group.value)
    if missing_groups:
        warnings.append(
            WarningItem(
                code="mapping.groups_without_data",
                message=(
                    "После применения mapping следующие группы не получили "
                    "наблюдений: " + ", ".join(missing_groups) + "."
                ),
                severity=WarningSeverity.WARNING,
                context={"groups": missing_groups},
            )
        )

    if not config.sheets:
        warnings.append(
            WarningItem(
                code="mapping.empty_config",
                message="Column mapping пустой: ни для одного листа не задано ролей.",
                severity=WarningSeverity.WARNING,
            )
        )

    if status == AnalysisStatus.NEEDS_COLUMN_MAPPING:
        warnings.append(
            WarningItem(
                code="mapping.no_zap_values",
                message=(
                    "После применения mapping ни одна ЗАП-колонка не дала "
                    "валидных значений. Проверьте выбранные колонки для роли 'ЗАП'."
                ),
                severity=WarningSeverity.WARNING,
            )
        )

    return warnings


def _build_hmm_charts(hmm: HMMResult) -> list[ChartData]:
    charts: list[ChartData] = []

    charts.append(
        ChartData(
            id="hmm_transition_matrix",
            title="Матрица переходов HMM",
            kind="heatmap",
            x=hmm.parameters.state_labels,
            y=hmm.parameters.state_labels,
            series=[{"matrix": hmm.parameters.transition_matrix}],
            meta={
                "n_states": hmm.parameters.n_states,
                "log_likelihood": hmm.parameters.log_likelihood,
                "converged": hmm.parameters.converged,
            },
        )
    )

    items = list(hmm.state_distribution.items())
    charts.append(
        ChartData(
            id="hmm_state_distribution",
            title="Распределение времени по скрытым состояниям",
            kind="bar",
            x=[k for k, _ in items],
            y=[v for _, v in items],
            meta={"group": "hmm_states"},
        )
    )

    confidences = [
        tr.confidence
        for tr in hmm.trajectories
        if tr.confidence is not None and math.isfinite(tr.confidence)
    ]
    if confidences:
        # Гистограмма confidence по эпизодам — 10 равных бакетов от 0 до 1.
        # Используется и как картинка, и как сигнал «много низких» в UI.
        bins = [i / 10 for i in range(11)]
        counts = [0] * 10
        for c in confidences:
            idx = min(int(c * 10), 9)
            counts[idx] += 1
        labels = [f"{bins[i]:.1f}–{bins[i + 1]:.1f}" for i in range(10)]
        charts.append(
            ChartData(
                id="hmm_confidence_histogram",
                title="Распределение уверенности Viterbi-пути (mean γ)",
                kind="bar",
                x=labels,
                y=counts,
                meta={
                    "group": "hmm_states",
                    "average_confidence": hmm.average_confidence,
                    "n_trajectories": len(confidences),
                },
            )
        )

    return charts


def _zap_candidate_unmapped_warnings(
    config: ColumnMappingConfig,
    detection: ColumnDetectionReport,
) -> list[WarningItem]:
    """Warning, если эвристика нашла ZAP-кандидата, а в mapping он не разметлен.

    Применяется ко всем кандидатам выше weak-порога (``score >= 0.5``):
    эвристика считает их потенциальной ZAP-наблюдаемой по DOMAIN_SPEC,
    но финальное решение остаётся за оператором. Если кандидат не вошёл
    в роль ZAP (включая случаи, когда оператор отменил preflight или
    собрал mapping вручную), оператор должен это увидеть — иначе
    плотность ZAP-сигнала может рухнуть незаметно.
    """

    out: list[WarningItem] = []
    for cand in detection.candidates:
        if cand.group != HiddenGroup.ZAP:
            continue
        if cand.score < 0.5:
            continue
        sm = config.sheets.get(cand.sheet)
        zap_cols = sm.roles.get(HiddenGroup.ZAP, []) if sm is not None else []
        if cand.column in zap_cols:
            continue
        out.append(
            WarningItem(
                code="mapping.zap_candidate_unmapped",
                message=(
                    f"На листе '{cand.sheet}' колонка '{cand.column}' "
                    "похожа на ZAP-наблюдение (по DOMAIN_SPEC), но в mapping "
                    "не размечена в роль ЗАП. Если это судейская оценка / "
                    "балл / удержание / болевой — добавьте её в роль ЗАП "
                    "через сопоставление колонок; иначе проигнорируйте."
                ),
                severity=WarningSeverity.WARNING,
                context={
                    "sheet": cand.sheet,
                    "column": cand.column,
                    "score": cand.score,
                    "rationale": cand.rationale,
                },
            )
        )
    return out


def _drop_totals_rows_from_frames(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Удалить строки-итоги из mapped frames до baseline/HMM.

    Аудит детектит итоги по сырому df (header_rows=[0]); здесь
    повторяем детектор уже на mapped frame, потому что после
    применения header_rows позиции строк смещены (заголовок съедает
    первые ``len(header_rows)`` строк) и могут различаться по числу
    данных. Возвращаем новый словарь frames и список словарей-контекстов
    для warning ``audit.totals_row_detected``.
    """

    new_frames: dict[str, pd.DataFrame] = {}
    info: list[dict[str, Any]] = []
    for sheet_name, df in frames.items():
        positions = detect_totals_row_indices(df)
        if not positions:
            new_frames[sheet_name] = df
            continue
        cleaned = drop_totals_rows(df, positions)
        new_frames[sheet_name] = cleaned
        info.append(
            {
                "sheet": sheet_name,
                "n_rows_dropped": len(positions),
                "rows_before": int(len(df)),
                "rows_after": int(len(cleaned)),
            }
        )
    return new_frames, info


def _try_apply_virtual_bout(
    config: ColumnMappingConfig,
    frames: dict[str, pd.DataFrame],
) -> tuple[
    ColumnMappingConfig,
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Попытаться добавить виртуальный bout по сбросам нумерации эпизодов.

    Не меняет переданные ``config``/``frames``: возвращает их новые
    in-memory копии. Виртуальный bout применяется к листу только
    если:

    * на листе нет роли ``bout`` (или она ссылается только на
      отсутствующие колонки);
    * присутствуют роли ``athlete`` и ``episode`` с реальными
      колонками в df;
    * :func:`hpc_algo.mapping.virtual_bout_series` вернула non-None;
    * медианная длина последовательности после группировки
      ``athlete + virtual_bout`` ≥ 2 (sanity-guard, чтобы виртуальный
      bout не вырождал HMM в траектории длины 1).

    Возвращает ``(new_config, new_frames, applied, missing)``:

    * ``applied`` — список словарей с контекстом для warning
      ``hmm.bout_inferred_from_episode_resets``.
    * ``missing`` — список словарей для warning ``mapping.bout_missing``
      (включая случаи отказа по sanity-guard).
    """

    new_config = config.model_copy(deep=True)
    new_frames: dict[str, pd.DataFrame] = {k: v.copy() for k, v in frames.items()}
    applied: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for sheet_name, sm in list(new_config.sheets.items()):
        df = new_frames.get(sheet_name)
        if df is None or df.empty:
            continue

        bout_cols = [
            c for c in (sm.roles.get(HiddenGroup.BOUT) or []) if c in df.columns
        ]
        if bout_cols:
            continue  # пользователь уже разметил bout — не вмешиваемся

        athlete_cols = [
            c for c in (sm.roles.get(HiddenGroup.ATHLETE) or []) if c in df.columns
        ]
        episode_cols = [
            c for c in (sm.roles.get(HiddenGroup.EPISODE) or []) if c in df.columns
        ]
        if not athlete_cols or not episode_cols:
            missing.append(
                {
                    "sheet": sheet_name,
                    "reason": "no_athlete_or_episode",
                    "has_athlete": bool(athlete_cols),
                    "has_episode": bool(episode_cols),
                }
            )
            continue

        athlete_col = athlete_cols[0]
        episode_col = episode_cols[0]
        vb = mapping_mod.virtual_bout_series(df, athlete_col, episode_col)
        if vb is None:
            missing.append(
                {
                    "sheet": sheet_name,
                    "reason": "episode_not_numeric",
                    "athlete_col": athlete_col,
                    "episode_col": episode_col,
                }
            )
            continue

        # Sanity-guard: считаем провизорные ключи и медиану длины серии
        # ДО фактического применения. Если меньше 2 — откат и mapping.bout_missing.
        # ФИО в merged-cell файлах заполнено только на первой строке блока — ffill.
        ath_filled = df[athlete_col].ffill()
        provisional_keys: list[str] = []
        for idx in df.index:
            ath_val = ath_filled.at[idx]
            ath_str = "" if pd.isna(ath_val) else str(ath_val).strip()
            if not ath_str:
                provisional_keys.append("")
                continue
            provisional_keys.append(f"{ath_str}|vb{int(vb.at[idx])}")
        counts: dict[str, int] = {}
        for key in provisional_keys:
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            missing.append(
                {
                    "sheet": sheet_name,
                    "reason": "no_valid_keys",
                    "athlete_col": athlete_col,
                    "episode_col": episode_col,
                }
            )
            continue
        lengths = sorted(counts.values())
        n = len(lengths)
        median_len = (
            lengths[n // 2]
            if n % 2 == 1
            else (lengths[n // 2 - 1] + lengths[n // 2]) // 2
        )
        if median_len < 2:
            missing.append(
                {
                    "sheet": sheet_name,
                    "reason": "median_too_short",
                    "athlete_col": athlete_col,
                    "episode_col": episode_col,
                    "median_sequence_length": int(median_len),
                    "n_sequences": int(n),
                }
            )
            continue

        col_name = _VIRTUAL_BOUT_COLUMN
        suffix = 1
        while col_name in df.columns:
            suffix += 1
            col_name = f"{_VIRTUAL_BOUT_COLUMN}{suffix}"
        df[col_name] = vb.values
        # ФИО хранится в merged-cells: forward-fill восстанавливает
        # принадлежность всех строк конкретному спортсмену, иначе HMM
        # собирает ключи group по неполным данным.
        df[athlete_col] = df[athlete_col].ffill()
        new_frames[sheet_name] = df
        new_roles: dict[HiddenGroup, list[str]] = {
            role: list(cols) for role, cols in sm.roles.items()
        }
        new_roles[HiddenGroup.BOUT] = [col_name]
        new_config.sheets[sheet_name] = SheetMapping(
            header_rows=list(sm.header_rows),
            data_start_row=sm.data_start_row,
            roles=new_roles,
        )
        applied.append(
            {
                "sheet": sheet_name,
                "athlete_col": athlete_col,
                "episode_col": episode_col,
                "virtual_bout_column": col_name,
                "n_virtual_bouts": int(
                    len({int(v) for v in vb.dropna().tolist()})
                ),
                "median_sequence_length": int(median_len),
                "n_sequences": int(n),
            }
        )

    return new_config, new_frames, applied, missing


def _analyze_with_mapping(
    loaded: LoadedExcel,
    audit: AuditReport,
    detection: ColumnDetectionReport,
    config: ColumnMappingConfig,
    analyze_config: AnalyzeConfig,
) -> AnalysisResult:
    frames = mapping_mod.load_mapped_sheets(loaded.path, config)
    frames, totals_filtered = _drop_totals_rows_from_frames(frames)
    config, frames, virtual_bout_applied, virtual_bout_missing = (
        _try_apply_virtual_bout(config, frames)
    )
    # Сужаем audit к листам, фактически вошедшим в анализ. Сам baseline
    # внутри уже фильтрует пропуски по mapping, но data_audit/отчёт/метаданные
    # должны быть согласованы с этим срезом.
    mapping_sheet_names = list(config.sheets.keys())
    audit_scoped = filter_audit_to_sheets(audit, set(mapping_sheet_names))
    baseline, unknown = build_baseline_with_mapping(frames, audit_scoped, config)
    status = _decide_status_with_mapping(baseline)

    # На mapping-пути не эмитим audit.possible_multirow_header: пользователь
    # уже задал header_rows в мастере, и подсказка про "укажите column_mapping
    # с header_rows" дублирует уже сделанный шаг.
    warnings: list[WarningItem] = []
    warnings.extend(_mapping_warnings(config, baseline, unknown, status))

    for info in virtual_bout_applied:
        warnings.append(
            WarningItem(
                code="hmm.bout_inferred_from_episode_resets",
                message=(
                    f"На листе '{info['sheet']}' колонка «Схватка» не размечена; "
                    "виртуальный bout восстановлен по сбросам нумерации эпизодов "
                    f"внутри одного борца ({info['n_virtual_bouts']} виртуальных "
                    f"схваток, медианная длина серии "
                    f"{info['median_sequence_length']}). Если разметка "
                    "«Схватка» доступна в исходных данных, добавьте её через "
                    "сопоставление колонок — это надёжнее."
                ),
                severity=WarningSeverity.INFO,
                context=info,
            )
        )

    for info in virtual_bout_missing:
        warnings.append(
            WarningItem(
                code="mapping.bout_missing",
                message=(
                    f"На листе '{info['sheet']}' нет роли «Схватка» (bout), и "
                    "виртуальный bout по сбросам нумерации эпизодов не "
                    "применён. Без bout HMM считает все эпизоды одного борца "
                    "одной серией, что может искажать структуру переходов. "
                    "Если в файле есть колонка «Схватка» — разметьте её через "
                    "сопоставление колонок; иначе проверьте, что нумерация "
                    "эпизодов перезапускается в каждой схватке."
                ),
                severity=WarningSeverity.INFO,
                context=info,
            )
        )

    for info in totals_filtered:
        warnings.append(
            WarningItem(
                code="audit.totals_row_detected",
                message=(
                    f"На листе '{info['sheet']}' распознано "
                    f"{info['n_rows_dropped']} строк-итогов "
                    "(маркеры «Итого/Всего/Сумма/Среднее/Total»). Они "
                    "исключены из baseline/HMM, чтобы суммы не "
                    "интерпретировались как отдельные эпизоды. "
                    "Если это нормальные данные — переименуйте такие "
                    "строки в исходном файле."
                ),
                severity=WarningSeverity.INFO,
                context=info,
            )
        )

    warnings.extend(_zap_candidate_unmapped_warnings(config, detection))

    # --- HMM-ветка (TASK_SPEC_004 / TASK_SPEC_005 / TASK_SPEC_008) ---
    hmm_result: HMMResult | None = None
    hmm_charts: list[ChartData] = []
    low_signal = False
    if status == AnalysisStatus.BASELINE_ONLY:
        run_cfg = analyze_config.hmm_run_config()
        sequences, alphabet = hmm_mod.build_observation_sequences(frames, config)
        guard_warnings = hmm_mod.evaluate_guards(
            baseline, config, sequences, alphabet, run_cfg
        )
        # Жёсткие guard'ы блокируют запуск, мягкие (`hmm.low_zap_density`)
        # — нет: их пробрасываем в warnings, но fit всё равно стартует.
        blocking = [w for w in guard_warnings if w.code == "hmm.guards_failed"]
        soft = [w for w in guard_warnings if w.code != "hmm.guards_failed"]
        warnings.extend(soft)
        low_signal = any(w.code == "hmm.low_zap_density" for w in soft)
        if blocking:
            warnings.extend(blocking)
        elif run_cfg.enable_hmm:
            if run_cfg.observation_emission == "bernoulli":
                fit_result = hmm_mod.fit_hmm_bernoulli(frames, config, run_cfg)
            else:
                fit_result = hmm_mod.fit_hmm(
                    sequences, alphabet, run_cfg, baseline=baseline
                )
            if fit_result is None:
                warnings.append(
                    WarningItem(
                        code="hmm.fit_failed",
                        message=(
                            "HMM не запустилась: не удалось обучить модель, "
                            "BIC-гейт/sanity не пройдены, или отсутствует "
                            "зависимость 'hmmlearn'. Статус остаётся baseline_only."
                        ),
                        severity=WarningSeverity.WARNING,
                        context={
                            "mode": run_cfg.mode,
                            "observation_emission": run_cfg.observation_emission,
                        },
                    )
                )
            else:
                result_obj, _sanity = fit_result
                hmm_result = result_obj
                hmm_charts = _build_hmm_charts(result_obj)
                status = (
                    AnalysisStatus.HMM_LOW_SIGNAL
                    if low_signal
                    else AnalysisStatus.HMM_READY
                )

    source_meta = SourceMetadata(
        filename=loaded.path.name,
        size_bytes=loaded.size_bytes,
        sha256=loaded.sha256,
        sheet_count=len(mapping_sheet_names),
        sheet_names=mapping_sheet_names,
    )
    charts = build_charts(baseline) + hmm_charts
    report_text = _build_report(
        status=status,
        source=source_meta,
        audit=audit_scoped,
        detection=detection,
        baseline=baseline,
        mapping_applied=True,
        hmm=hmm_result,
    )

    trainer_summary = build_athlete_episode_rollup(frames, config)

    return AnalysisResult(
        status=status,
        source_metadata=source_meta,
        data_audit=audit_scoped,
        detected_columns=detection,
        basic_statistics=baseline,
        applied_mapping=config,
        hmm=hmm_result,
        charts=charts,
        warnings=warnings,
        errors=[],
        trainer_athlete_summary=trainer_summary,
        report=report_text,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_source(
    source_path: str | Path,
    config: AnalyzeConfig | None = None,
) -> AnalysisResult:
    """Проанализировать Excel-источник.

    Если в ``config.column_mapping`` передан подтверждённый
    :class:`ColumnMappingConfig`, анализ идёт по mapping-пути и способен
    вернуть ``baseline_only`` с реальными распределениями по всем 4 группам.
    """

    path = Path(source_path)
    size_bytes = path.stat().st_size if path.exists() and path.is_file() else 0

    try:
        loaded = load_excel(path)
    except ExcelLoadError as exc:
        return _failed_result(
            path,
            size_bytes,
            WarningItem(
                code="loading.excel_error",
                message=str(exc),
                severity=WarningSeverity.ERROR,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_result(
            path,
            size_bytes,
            WarningItem(
                code="loading.unexpected_error",
                message=f"Неожиданная ошибка при чтении Excel: {exc}",
                severity=WarningSeverity.ERROR,
            ),
        )

    audit = build_audit(loaded)
    detection = detect_columns(loaded)

    if config is None:
        config = AnalyzeConfig()

    if (
        config.column_mapping is not None
        and not config.column_mapping.is_empty()
    ):
        return _analyze_with_mapping(
            loaded, audit, detection, config.column_mapping, config
        )

    baseline = build_baseline(loaded, audit, detection)
    charts = build_charts(baseline)
    status = _decide_status_heuristic(detection, audit)
    warnings = _build_heuristic_warnings(detection, audit, status)

    source_meta = SourceMetadata(
        filename=loaded.path.name,
        size_bytes=loaded.size_bytes,
        sha256=loaded.sha256,
        sheet_count=len(loaded.sheet_names),
        sheet_names=loaded.sheet_names,
    )

    report_text = _build_report(
        status=status,
        source=source_meta,
        audit=audit,
        detection=detection,
        baseline=baseline,
        mapping_applied=False,
    )

    return AnalysisResult(
        status=status,
        source_metadata=source_meta,
        data_audit=audit,
        detected_columns=detection,
        basic_statistics=baseline,
        charts=charts,
        warnings=warnings,
        report=report_text,
    )


def preflight_mapping(
    source_path: str | Path,
    sheet_names: list[str] | None = None,
) -> ColumnMappingConfig:
    """Вернуть предполагаемый :class:`ColumnMappingConfig`.

    Функция нейтральная: эвристически угадывает ``header_rows``,
    делает flatten колонок и предлагает роли. Пользователь обязан
    подтвердить результат перед использованием.

    ``sheet_names`` — опциональный whitelist листов (используется мастером
    предобработки, когда пользователь явно исключил часть листов из
    анализа). Если ``None`` — preflight проходит по всем листам файла.
    """

    path = Path(source_path)
    return mapping_mod.preflight(path, sheet_names=sheet_names)


def list_workbook_sheets(source_path: str | Path) -> list[str]:
    """Вернуть список листов Excel-файла без полного парсинга содержимого."""

    path = Path(source_path)
    xl = pd.ExcelFile(path, engine="openpyxl")
    return list(xl.sheet_names)


def analysis_summary(result: AnalysisResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "filename": result.source_metadata.filename,
        "sheets": result.source_metadata.sheet_names,
        "total_rows": result.data_audit.total_rows,
        "detected_groups": [g.value for g in result.detected_columns.detected_groups],
        "missing_groups": [g.value for g in result.detected_columns.missing_groups],
        "zap_candidates": [
            {"sheet": c.sheet, "column": c.column, "score": c.score}
            for c in result.detected_columns.candidates
            if c.group == HiddenGroup.ZAP
        ],
        "hidden_group_totals": result.basic_statistics.hidden_group_totals,
        "applied_mapping": result.applied_mapping is not None,
        "warnings": [w.code for w in result.warnings],
        "errors": [e.code for e in result.errors],
    }


def is_honest_baseline(result: AnalysisResult) -> bool:
    return result.status in {
        AnalysisStatus.AUDIT_ONLY,
        AnalysisStatus.BASELINE_ONLY,
        AnalysisStatus.NEEDS_COLUMN_MAPPING,
        AnalysisStatus.HMM_LOW_SIGNAL,
        AnalysisStatus.FAILED,
    }


__all__ = [
    "AnalyzeConfig",
    "analyze_source",
    "preflight_mapping",
    "analysis_summary",
    "is_honest_baseline",
]
