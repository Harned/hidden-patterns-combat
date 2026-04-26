"""Публичный интерфейс processing module.

Главная точка входа — :func:`analyze_source`. Возвращает
:class:`AnalysisResult` и никогда не бросает исключений наверх: любая
проблема оказывается в поле ``errors`` результата.

Также доступен :func:`preflight_mapping` — safe-to-call функция,
возвращающая предлагаемый :class:`ColumnMappingConfig` для данного файла.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hpc_algo import hmm as hmm_mod
from hpc_algo import mapping as mapping_mod
from hpc_algo.audit import build_audit
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
    SourceMetadata,
    WarningItem,
    WarningSeverity,
)


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
) -> str:
    lines: list[str] = []
    lines.append(f"Файл: {source.filename}")
    lines.append(f"Листов: {source.sheet_count}. Всего строк: {audit.total_rows}.")
    lines.append(f"Доля пропусков по файлу: {audit.overall_null_ratio * 100:.1f}%.")

    if mapping_applied:
        totals = baseline.hidden_group_totals
        if totals:
            lines.append(
                "Наблюдения по группам: "
                + ", ".join(f"{k}={v}" for k, v in totals.items())
                + "."
            )
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
        AnalysisStatus.FAILED: "Статус: failed — см. errors.",
    }
    lines.append(status_messages.get(status, f"Статус: {status.value}."))

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

    # transition heatmap
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

    # state distribution (bar)
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

    return charts


def _analyze_with_mapping(
    loaded: LoadedExcel,
    audit: AuditReport,
    detection: ColumnDetectionReport,
    config: ColumnMappingConfig,
    analyze_config: AnalyzeConfig,
) -> AnalysisResult:
    frames = mapping_mod.load_mapped_sheets(loaded.path, config)
    baseline, unknown = build_baseline_with_mapping(frames, audit, config)
    status = _decide_status_with_mapping(baseline)

    warnings: list[WarningItem] = []
    mr = _build_multirow_warning(audit)
    if mr is not None:
        warnings.append(mr)
    warnings.extend(_mapping_warnings(config, baseline, unknown, status))

    # --- HMM-ветка (TASK_SPEC_004 / TASK_SPEC_005 / TASK_SPEC_008) ---
    hmm_result: HMMResult | None = None
    hmm_charts: list[ChartData] = []
    if status == AnalysisStatus.BASELINE_ONLY:
        run_cfg = analyze_config.hmm_run_config()
        sequences, alphabet = hmm_mod.build_observation_sequences(frames, config)
        guard_warnings = hmm_mod.evaluate_guards(
            baseline, config, sequences, alphabet, run_cfg
        )
        if guard_warnings:
            warnings.extend(guard_warnings)
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
                status = AnalysisStatus.HMM_READY

    source_meta = SourceMetadata(
        filename=loaded.path.name,
        size_bytes=loaded.size_bytes,
        sha256=loaded.sha256,
        sheet_count=len(loaded.sheet_names),
        sheet_names=loaded.sheet_names,
    )
    charts = build_charts(baseline) + hmm_charts
    report_text = _build_report(
        status=status,
        source=source_meta,
        audit=audit,
        detection=detection,
        baseline=baseline,
        mapping_applied=True,
    )

    return AnalysisResult(
        status=status,
        source_metadata=source_meta,
        data_audit=audit,
        detected_columns=detection,
        basic_statistics=baseline,
        applied_mapping=config,
        hmm=hmm_result,
        charts=charts,
        warnings=warnings,
        errors=[],
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

    import pandas as pd

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
        AnalysisStatus.FAILED,
    }


__all__ = [
    "AnalyzeConfig",
    "analyze_source",
    "preflight_mapping",
    "analysis_summary",
    "is_honest_baseline",
]
