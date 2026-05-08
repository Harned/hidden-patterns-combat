"""Оркестратор: Excel + YAML → ~30 индивидуальных HTML-отчётов.

Ключевая публичная функция — :func:`build_individual_models`. CLI и
скрипт ``scripts/build_individual_models.py`` делают только парсинг
аргументов и вызывают её.

Пайплайн на каждый запуск:

1. Загрузить ``config/state_groups.yaml`` (`load_state_groups`).
2. Прочитать лист (по умолчанию `Общее`) с 3-уровневой шапкой и
   привести имена колонок к flatten-форме.
3. Сверить колонки YAML с реально присутствующими.
4. Определить служебные колонки (``athlete``, ``№ эпизода``, ...) и
   собрать список feature-колонок (объединение по всем не-pause состояниям).
5. Разрезать DataFrame на bout'ы / эпизоды.
6. Построить EpisodeRecord-последовательность.
7. Для каждого спортсмена: ``fit_individual_markov`` →
   ``compute_episode_metrics`` → ``render_individual_report``
   → запись в ``output_dir/<slug>.html``.
8. Сгенерировать ``index.html`` со ссылками и ``summary.json``.

Никаких HMM-зависимостей. Никаких сетевых ресурсов в HTML.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hpc_algo.episode_metrics import (
    classify_style,
    compute_episode_metrics,
    load_style_thresholds,
)
from hpc_algo.episode_split import (
    detect_base_columns,
    read_episodes_sheet,
    split_into_bouts_and_episodes,
)
from hpc_algo.markov_individual import build_episode_sequence, fit_individual_markov
from hpc_algo.report_individual import render_index_page, render_individual_report
from hpc_algo.schema import (
    BuildIndividualSummary,
    EpisodeState,
    MarkovWarning,
)
from hpc_algo.state_groups import (
    StateGroupsConfig,
    load_state_groups,
    validate_columns_against_sheet,
)


@dataclass(frozen=True)
class _AthleteOutput:
    athlete: str
    slug: str
    html_path: Path
    warnings_count: int


def _slugify(name: str) -> str:
    """Превратить ФИО в безопасный для FS идентификатор.

    Кириллица сохраняется (macOS/Linux корректно работают с UTF-8 в FS),
    пробелы → ``_``, всё, что не ``\\w`` или ``-`` после strip,
    отбрасывается. Если имя «выпало» в пустую строку — fallback на
    ``athlete``.
    """

    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-]", "", s, flags=re.UNICODE)
    return s or "athlete"


def _feature_columns(cfg: StateGroupsConfig) -> list[str]:
    """Объединение всех колонок по не-pause состояниям, сохраняя порядок."""

    out: list[str] = []
    seen: set[str] = set()
    for state, cols in cfg.states.items():
        if state == EpisodeState.PAUSE.value:
            continue
        for col in cols:
            if col not in seen:
                seen.add(col)
                out.append(col)
    return out


_NON_ATHLETE_HINTS: tuple[str, ...] = ("итог", "всего", "сумма", "total")


def _is_non_athlete_label(name: str) -> bool:
    n = name.strip().lower()
    return any(h in n for h in _NON_ATHLETE_HINTS)


def _athletes_in_order(records: list, raw_episodes: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for source in (records, raw_episodes):
        for r in source:
            name = getattr(r, "athlete", None)
            if not name or not name.strip():
                continue
            if _is_non_athlete_label(name):
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def build_individual_models(
    excel_path: str | Path,
    state_groups_path: str | Path,
    output_dir: str | Path,
    *,
    sheet: str | None = None,
    athlete_filter: list[str] | None = None,
    style_thresholds_path: str | Path | None = None,
    header_rows: tuple[int, ...] | None = None,
) -> BuildIndividualSummary:
    """Собрать индивидуальные HTML-отчёты для всех (или выбранных) спортсменов.

    Если ``style_thresholds_path`` указан, дополнительно классифицируем
    стиль через :func:`hpc_algo.episode_metrics.classify_style` и
    встраиваем результат в HTML. Без порогов отчёт строится без блока
    «Стиль» и без warning'а: классификация — описательное расширение,
    необязательное.
    """

    excel_path = Path(excel_path)
    state_groups_path = Path(state_groups_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, cfg_warnings = load_state_groups(state_groups_path)
    target_sheet = sheet or cfg.sheet

    style_thresholds: dict[str, dict[str, float]] | None = None
    style_thresholds_warnings: list[MarkovWarning] = []
    if style_thresholds_path is not None:
        style_thresholds, style_thresholds_warnings = load_style_thresholds(
            style_thresholds_path
        )

    df = read_episodes_sheet(excel_path, sheet=target_sheet, header_rows=header_rows)
    available_columns = list(df.columns)

    column_validation_warnings = validate_columns_against_sheet(cfg, available_columns)

    base = detect_base_columns(available_columns)
    feature_cols = _feature_columns(cfg)

    raw_episodes, split_warnings = split_into_bouts_and_episodes(
        df, base, feature_cols
    )
    records = build_episode_sequence(raw_episodes, cfg)

    athletes = _athletes_in_order(records, raw_episodes)
    if athlete_filter is not None:
        wanted = {a.strip() for a in athlete_filter if a and a.strip()}
        athletes = [a for a in athletes if a in wanted]

    rendered: list[_AthleteOutput] = []
    skipped: list[str] = []

    for athlete in athletes:
        result = fit_individual_markov(athlete, records, mode=cfg.mode)
        if result.episode_count == 0:
            skipped.append(athlete)
            continue
        metrics = compute_episode_metrics(raw_episodes, records, athlete=athlete)
        athlete_warnings_extra: list[MarkovWarning] = []
        if style_thresholds is not None:
            label, style_warning = classify_style(metrics, style_thresholds)
            metrics = metrics.model_copy(update={"style": label})
            if style_warning is not None:
                athlete_warnings_extra.append(style_warning)
        html_text = render_individual_report(
            result.model_copy(
                update={
                    "warnings": list(result.warnings) + athlete_warnings_extra,
                }
            ),
            metrics,
        )
        slug = _slugify(athlete)
        html_path = output_dir / f"{slug}.html"
        html_path.write_text(html_text, encoding="utf-8")
        rendered.append(
            _AthleteOutput(
                athlete=athlete,
                slug=slug,
                html_path=html_path,
                warnings_count=len(result.warnings) + len(athlete_warnings_extra),
            )
        )

    summary_lines = [
        f"Источник: {excel_path.name} (лист «{target_sheet}»)",
        f"Конфиг состояний: {state_groups_path.name}",
        f"Эпизодов всего: {len(raw_episodes)}; "
        f"спортсменов с моделью: {len(rendered)}; пропущено: {len(skipped)}",
    ]
    index_html = render_index_page(
        [(o.athlete, f"{o.slug}.html") for o in rendered],
        generated_at=datetime.utcnow(),
        summary_lines=summary_lines,
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    summary = BuildIndividualSummary(
        source=str(excel_path),
        sheet=target_sheet,
        state_groups_path=str(state_groups_path),
        output_dir=str(output_dir),
        athletes_total=len(athletes),
        athletes_rendered=len(rendered),
        skipped_athletes=skipped,
        rendered_athletes=[o.athlete for o in rendered],
        config_warnings=cfg_warnings + style_thresholds_warnings,
        split_warnings=_dedupe_warnings(split_warnings),
        column_validation_warnings=column_validation_warnings,
        per_athlete_warning_counts={o.athlete: o.warnings_count for o in rendered},
    )
    summary_json = json.loads(summary.model_dump_json())
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def _dedupe_warnings(warnings: list[MarkovWarning]) -> list[MarkovWarning]:
    """Свернуть повторяющиеся data.value_out_of_range / non_numeric в один.

    Без агрегации список из нескольких сотен одинаковых warning'ов на
    реальных Excel'ях делает summary.json нечитаемым. Сохраняем количество
    в ``context["occurrences"]`` и одно сообщение на ``code``.
    """

    if not warnings:
        return []
    by_code: dict[str, list[MarkovWarning]] = {}
    for w in warnings:
        by_code.setdefault(w.code, []).append(w)
    out: list[MarkovWarning] = []
    for code, items in by_code.items():
        if code in {"data.value_out_of_range", "data.non_numeric_feature"} and len(items) > 1:
            first = items[0]
            out.append(
                MarkovWarning(
                    code=code,
                    message=f"{first.message} (агрегировано {len(items)} вхождений)",
                    context={
                        "occurrences": len(items),
                        "first_context": first.context,
                    },
                )
            )
        else:
            out.extend(items)
    return out
