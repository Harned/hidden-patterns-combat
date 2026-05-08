"""Оркестратор: Excel + YAML(state-groups) + finalists → 10 агрегатных HTML.

Один прогон ``make aggregate-models`` собирает по одному отчёту на
весовую категорию: суммирует ``transition_counts`` индивидуалов
призёров 1–3, считает дивергенцию каждого индивидуала относительно
агрегата, ранжирует, рендерит самодостаточный HTML.

Список призёров берётся из ``config/finalists.yaml``. Если файла нет,
делается осторожная попытка извлечь его из колонок
``Весовая категория`` / ``Место`` — это fallback, не основной путь.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from hpc_algo.build_individual import _slugify
from hpc_algo.compare import rank_within_class
from hpc_algo.episode_split import (
    detect_base_columns,
    read_episodes_sheet,
    split_into_bouts_and_episodes,
)
from hpc_algo.finalists import finalists_from_episodes, load_finalists_yaml
from hpc_algo.markov_aggregate import fit_aggregate_markov
from hpc_algo.markov_individual import build_episode_sequence, fit_individual_markov
from hpc_algo.report_aggregate import render_aggregate_index, render_aggregate_report
from hpc_algo.schema import (
    BuildAggregateSummary,
    EpisodeState,
    FinalistEntry,
    MarkovWarning,
)
from hpc_algo.state_groups import (
    StateGroupsConfig,
    load_state_groups,
    validate_columns_against_sheet,
)


def _feature_columns(cfg: StateGroupsConfig) -> list[str]:
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


def _resolve_finalists(
    finalists_path: str | Path | None,
    raw_episodes: list,
    df_rows: list[dict],
    df_columns: list[str],
) -> tuple[list[FinalistEntry], list[MarkovWarning]]:
    """YAML приоритетнее эвристического детекта по Excel."""

    if finalists_path is not None:
        return load_finalists_yaml(finalists_path)
    return finalists_from_episodes(raw_episodes, df_columns, df_rows)


def build_aggregate_models(
    excel_path: str | Path,
    state_groups_path: str | Path,
    output_dir: str | Path,
    *,
    finalists_path: str | Path | None = None,
    sheet: str | None = None,
    alpha: float = 0.5,
    header_rows: tuple[int, ...] | None = None,
) -> BuildAggregateSummary:
    """Собрать 10 агрегатных HTML-отчётов для весовых категорий."""

    excel_path = Path(excel_path)
    state_groups_path = Path(state_groups_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    finalists_path_p = Path(finalists_path) if finalists_path is not None else None

    cfg, cfg_warnings = load_state_groups(state_groups_path)
    target_sheet = sheet or cfg.sheet

    df = read_episodes_sheet(excel_path, sheet=target_sheet, header_rows=header_rows)
    available_columns = list(df.columns)
    column_validation_warnings = validate_columns_against_sheet(cfg, available_columns)

    base = detect_base_columns(available_columns)
    feature_cols = _feature_columns(cfg)
    raw_episodes, split_warnings = split_into_bouts_and_episodes(
        df, base, feature_cols
    )
    records = build_episode_sequence(raw_episodes, cfg)

    df_rows: list[dict] = (
        df.to_dict(orient="records") if finalists_path_p is None else []
    )

    finalists, finalists_warnings = _resolve_finalists(
        finalists_path_p, raw_episodes, df_rows, available_columns
    )

    by_class: dict[str, list[FinalistEntry]] = defaultdict(list)
    for entry in finalists:
        # TS_012 § Предметные инварианты — агрегат строится только по
        # призёрам 1–3 каждой весовой категории. Прочих финалистов
        # игнорируем (но оставляем warning, чтобы не молча терять данные).
        if entry.place > 3:
            finalists_warnings.append(
                MarkovWarning(
                    code="aggregate.non_medalist_dropped",
                    message=(
                        f"Спортсмен {entry.athlete!r} с местом {entry.place} в"
                        f" категории {entry.weight_class!r} не попадает в"
                        " агрегат (агрегируем только 1–3)."
                    ),
                    context={
                        "athlete": entry.athlete,
                        "place": entry.place,
                        "weight_class": entry.weight_class,
                    },
                )
            )
            continue
        by_class[entry.weight_class].append(entry)

    rendered_classes: list[tuple[str, str]] = []
    skipped_classes: list[str] = []
    per_class_member_counts: dict[str, int] = {}

    for wc in sorted(by_class.keys()):
        entries = by_class[wc]
        if not entries:
            skipped_classes.append(wc)
            continue
        individuals = []
        athlete_to_place: dict[str, int] = {}
        for entry in entries:
            indiv = fit_individual_markov(entry.athlete, records, mode=cfg.mode)
            if indiv.episode_count == 0:
                continue
            individuals.append(indiv)
            athlete_to_place[entry.athlete] = entry.place
        if not individuals:
            skipped_classes.append(wc)
            continue

        aggregate = fit_aggregate_markov(individuals, weight_class=wc, mode=cfg.mode)
        ranking = rank_within_class(
            individuals, aggregate, alpha=alpha, places=athlete_to_place
        )
        html = render_aggregate_report(aggregate, ranking, alpha=alpha)
        slug = _slugify(wc)
        out_path = output_dir / f"{slug}.html"
        out_path.write_text(html, encoding="utf-8")
        rendered_classes.append((wc, f"{slug}.html"))
        per_class_member_counts[wc] = aggregate.members_count

    summary_lines = [
        f"Источник: {excel_path.name} (лист «{target_sheet}»)",
        f"Конфиг состояний: {state_groups_path.name}",
        f"Финалисты: {finalists_path_p.name if finalists_path_p else 'из Excel (heuristic)'}",
        f"Категорий с агрегатом: {len(rendered_classes)}; "
        f"пропущено: {len(skipped_classes)}",
        f"α композитной метрики: {alpha:.2f}",
    ]
    index_html = render_aggregate_index(
        rendered_classes,
        generated_at=datetime.utcnow(),
        summary_lines=summary_lines,
    )
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    summary = BuildAggregateSummary(
        source=str(excel_path),
        sheet=target_sheet,
        state_groups_path=str(state_groups_path),
        finalists_path=str(finalists_path_p) if finalists_path_p else None,
        output_dir=str(output_dir),
        weight_classes_total=len(by_class),
        weight_classes_rendered=len(rendered_classes),
        rendered_weight_classes=[wc for wc, _ in rendered_classes],
        skipped_weight_classes=skipped_classes,
        config_warnings=cfg_warnings,
        finalists_warnings=finalists_warnings,
        split_warnings=_dedupe_warnings(split_warnings),
        column_validation_warnings=column_validation_warnings,
        per_class_member_counts=per_class_member_counts,
        alpha=alpha,
    )
    summary_json = json.loads(summary.model_dump_json())
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _dedupe_warnings(warnings: list[MarkovWarning]) -> list[MarkovWarning]:
    """Те же кейсы, что в build_individual: схлопываем шумные коды."""

    if not warnings:
        return []
    by_code: dict[str, list[MarkovWarning]] = {}
    for w in warnings:
        by_code.setdefault(w.code, []).append(w)
    out: list[MarkovWarning] = []
    for code, items in by_code.items():
        if (
            code in {"data.value_out_of_range", "data.non_numeric_feature"}
            and len(items) > 1
        ):
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
