"""Самодостаточный HTML-отчёт по агрегатной модели весовой категории (TS_012).

Использует те же стилевые примитивы, что и
:mod:`hpc_algo.report_individual` (heatmap-ячейки HSL, π-bars,
warnings-list). Никаких CDN/JS.

Содержимое:

* heatmap агрегатной матрицы переходов;
* π-распределение агрегата;
* таблица ранжирования финалистов с колонками
  ``rank | athlete | place | l1_stationary | kl_transitions | composite``;
* список warnings агрегата.
"""

from __future__ import annotations

from datetime import datetime

from hpc_algo.report_individual import (
    _CSS,
    _esc,
    _fmt_float,
    _render_heatmap,
    _render_pi_bars,
    _render_visit_counts,
    _render_warnings,
)
from hpc_algo.schema import MarkovAggregateResult, RankingEntry


def _render_ranking(ranking: list[RankingEntry]) -> str:
    if not ranking:
        return '<p class="muted">Список ранжирования пуст.</p>'
    rows: list[str] = []
    for entry in ranking:
        d = entry.divergence
        place_text = "—" if entry.place is None else str(entry.place)
        rows.append(
            "<tr>"
            f"<td>{entry.rank}</td>"
            f'<td class="row-head">{_esc(entry.athlete)}</td>'
            f"<td>{_esc(place_text)}</td>"
            f"<td>{_fmt_float(d.l1_stationary)}</td>"
            f"<td>{_fmt_float(d.kl_transitions)}</td>"
            f"<td>{_fmt_float(d.composite)}</td>"
            "</tr>"
        )
    return (
        '<table class="kv-table"><thead><tr>'
        "<th>Ранг</th><th>Спортсмен</th><th>Место</th>"
        "<th>L1(π)</th><th>KL(A)</th><th>Composite</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_aggregate_report(
    result: MarkovAggregateResult,
    ranking: list[RankingEntry],
    *,
    state_labels_ru: dict[str, str] | None = None,
    title: str | None = None,
    generated_at: datetime | None = None,
    alpha: float | None = None,
) -> str:
    """Сформировать HTML-отчёт по агрегатной модели одной весовой категории."""

    from hpc_algo.report_individual import _STATE_LABELS_RU

    labels = state_labels_ru or _STATE_LABELS_RU
    page_title = title or f"Агрегатная Marков-цепь — категория {result.weight_class}"
    ts = (generated_at or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
    alpha_text = (
        f" · α = {alpha:.2f}"
        if alpha is not None
        else (
            f" · α = {ranking[0].divergence.alpha:.2f}"
            if ranking
            else ""
        )
    )

    body = [
        f"<h1>{_esc(page_title)}</h1>",
        f'<p class="meta">Категория: <b>{_esc(result.weight_class)}</b> · '
        f"режим: <code>{_esc(result.mode)}</code> · "
        f"спортсменов: {result.members_count}{alpha_text} · "
        f"эпизодов: {result.episode_count_total} · "
        f"поединков: {result.bout_count_total} · "
        f"сгенерировано: {_esc(ts)}</p>",
        "<section><h2>Состав агрегата</h2>",
        '<p class="muted">Только призёры 1–3 (по '
        "<code>config/finalists.yaml</code>); посторонние не попадают.</p>",
        "<ul>"
        + "".join(f"<li>{_esc(m)}</li>" for m in result.members)
        + "</ul></section>",
        "<section><h2>Матрица переходов A_aggregate</h2>",
        '<p class="muted">Считается через сумму transition_counts '
        "индивидуалов; ложные межатлетные переходы исключены.</p>",
        _render_heatmap(result.transition_matrix, list(result.states), labels),
        "</section>",
        "<section><h2>Стационарное распределение π_aggregate</h2>",
        _render_pi_bars(list(result.stationary), list(result.states), labels),
        "</section>",
        "<section><h2>Посещения состояний (суммарно)</h2>",
        _render_visit_counts(result.visit_counts, list(result.states), labels),
        "</section>",
        "<section><h2>Ранжирование финалистов категории</h2>",
        '<p class="muted">Чем меньше Composite, тем ближе индивидуал '
        "к «средней формуле победы среди призёров». Composite = "
        "α·L1(π) + (1−α)·KL(A); KL — асимметричная.</p>",
        _render_ranking(ranking),
        "</section>",
        "<section><h2>Предупреждения модели</h2>",
        _render_warnings(list(result.warnings)),
        "</section>",
    ]

    return (
        "<!doctype html>"
        '<html lang="ru"><head>'
        '<meta charset="utf-8">'
        f"<title>{_esc(page_title)}</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        + "".join(body)
        + "</body></html>"
    )


def render_aggregate_index(
    rendered: list[tuple[str, str]],
    *,
    title: str = "Агрегатные Marков-модели по призёрам",
    generated_at: datetime | None = None,
    summary_lines: list[str] | None = None,
) -> str:
    """Index-страница со ссылками на отчёты по весовым категориям."""

    ts = (generated_at or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
    items = "".join(
        f'<li><a href="{_esc(href)}">Категория {_esc(name)}</a></li>'
        for name, href in rendered
    )
    summary_block = ""
    if summary_lines:
        summary_block = (
            "<section><h2>Сводка</h2><ul>"
            + "".join(f"<li>{_esc(line)}</li>" for line in summary_lines)
            + "</ul></section>"
        )
    return (
        "<!doctype html>"
        '<html lang="ru"><head>'
        '<meta charset="utf-8">'
        f"<title>{_esc(title)}</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        f"<h1>{_esc(title)}</h1>"
        f'<p class="meta">Сгенерировано: {_esc(ts)}</p>'
        + summary_block
        + '<section><h2>Категории</h2><ul class="reports-list">'
        + items
        + "</ul></section>"
        "</body></html>"
    )
