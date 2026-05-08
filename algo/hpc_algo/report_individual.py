"""Самодостаточный HTML-отчёт по индивидуальной 5-state Marков-цепи.

Никаких CDN, никакого JavaScript. Inline CSS + чистый HTML5 ради
воспроизводимости и совместимости с офлайн-просмотром по требованию
``TASK_SPEC_011`` (DoD §1, §6).

Контракт: одна функция :func:`render_individual_report`, принимающая
:class:`MarkovIndividualResult` и :class:`EpisodeMetrics`, возвращающая
строку HTML. Никакого I/O — запись на диск делает оркестратор
(``build_individual.py``).
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

from hpc_algo.schema import (
    EpisodeMetrics,
    EpisodeState,
    MarkovIndividualResult,
)

_STATE_LABELS_RU: dict[str, str] = {
    EpisodeState.MANOEUVRING.value: "Маневрирование",
    EpisodeState.GRIP.value: "Захват / КФВ",
    EpisodeState.OFF_BALANCE.value: "Выведение из равновесия (ВУП)",
    EpisodeState.TECHNICAL_ACTION.value: "Завершающий приём (ЗАП)",
    EpisodeState.PAUSE.value: "Пауза / нет активности",
}


def _esc(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        if value != value:  # NaN
            return "—"
    except TypeError:
        return "—"
    return f"{value:.{digits}f}"


def _heatmap_cell_color(p: float) -> str:
    """Цвет ячейки: HSL от светлого к тёмному в синей гамме.

    p ∈ [0, 1]; saturation растёт с p, lightness падает с p.
    Никаких внешних библиотек.
    """

    p = max(0.0, min(1.0, float(p)))
    lightness = 95 - int(round(p * 55))
    saturation = 35 + int(round(p * 50))
    return f"hsl(210, {saturation}%, {lightness}%)"


def _bar_block(p: float) -> str:
    width_pct = max(0.0, min(1.0, p)) * 100.0
    return (
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{width_pct:.2f}%"></div>'
        f'<span class="bar-label">{_fmt_float(p)}</span>'
        "</div>"
    )


def _label(state_value: str, labels: dict[str, str]) -> str:
    return labels.get(state_value, state_value)


def _render_heatmap(
    matrix: list[list[float]],
    states: list[EpisodeState],
    labels: dict[str, str],
) -> str:
    head_cells = "".join(
        f'<th class="col-head">{_esc(_label(s.value, labels))}</th>' for s in states
    )
    rows: list[str] = []
    for i, row in enumerate(matrix):
        cells = "".join(
            f'<td class="hm-cell" style="background:{_heatmap_cell_color(p)}" '
            f'title="{_esc(states[i].value)} → {_esc(states[j].value)}: '
            f'{_fmt_float(p)}">{_fmt_float(p, 2)}</td>'
            for j, p in enumerate(row)
        )
        rows.append(
            f'<tr><th class="row-head">{_esc(_label(states[i].value, labels))}</th>{cells}</tr>'
        )
    return (
        '<table class="heatmap"><thead><tr>'
        '<th class="corner">из / в</th>'
        f"{head_cells}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_pi_bars(
    stationary: list[float],
    states: list[EpisodeState],
    labels: dict[str, str],
) -> str:
    rows = []
    for s, p in zip(states, stationary, strict=False):
        rows.append(
            "<tr>"
            f'<th class="row-head">{_esc(_label(s.value, labels))}</th>'
            f"<td>{_bar_block(p)}</td>"
            "</tr>"
        )
    return (
        '<table class="pi-table"><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_visit_counts(
    visit_counts: dict[str, int],
    states: list[EpisodeState],
    labels: dict[str, str],
) -> str:
    total = sum(visit_counts.values()) or 0
    rows: list[str] = []
    for s in states:
        c = int(visit_counts.get(s.value, 0))
        share = (c / total) if total > 0 else 0.0
        rows.append(
            "<tr>"
            f'<th class="row-head">{_esc(_label(s.value, labels))}</th>'
            f"<td>{c}</td>"
            f"<td>{_fmt_float(share)}</td>"
            "</tr>"
        )
    return (
        '<table class="kv-table"><thead><tr>'
        "<th>Состояние</th><th>Эпизодов</th><th>Доля</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_metrics(metrics: EpisodeMetrics) -> str:
    d = metrics.duration_stats
    p = metrics.pause_stats

    def _row(label: str, value: str) -> str:
        return f"<tr><th>{_esc(label)}</th><td>{value}</td></tr>"

    rows = [
        _row("Эпизодов всего", str(metrics.episode_count)),
        _row("Поединков (bout)", str(metrics.bout_count)),
        _row(
            "Длительность эпизода (среднее / медиана / std)",
            f"{_fmt_float(d.mean, 2)} / {_fmt_float(d.median, 2)} / {_fmt_float(d.std, 2)} с",
        ),
        _row(
            "Длительность эпизода (min / max / total)",
            f"{_fmt_float(d.min, 2)} / {_fmt_float(d.max, 2)} / {_fmt_float(d.total, 2)} с",
        ),
        _row(
            "Длительность паузы (среднее / медиана)",
            f"{_fmt_float(p.mean, 2)} / {_fmt_float(p.median, 2)} с",
        ),
        _row("Действий в секунду (action_density)", _fmt_float(metrics.action_density)),
        _row("Доля не-ЗАП эпизодов", _fmt_float(metrics.non_technical_share)),
        _row("Равномерность активности (entropy)", _fmt_float(metrics.activity_evenness)),
    ]
    return (
        '<table class="kv-table"><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_warnings(warnings: list[Any]) -> str:
    if not warnings:
        return '<p class="muted">Предупреждений нет.</p>'
    items: list[str] = []
    for w in warnings:
        code = _esc(getattr(w, "code", ""))
        message = _esc(getattr(w, "message", ""))
        items.append(f"<li><code>{code}</code> — {message}</li>")
    return f'<ul class="warnings">{"".join(items)}</ul>'


def _argmax_state(matrix_row: list[float]) -> int:
    best = 0
    best_v = -1.0
    for i, v in enumerate(matrix_row):
        if v > best_v:
            best_v = v
            best = i
    return best


def _build_interpretation(
    result: MarkovIndividualResult,
    metrics: EpisodeMetrics,
    labels: dict[str, str],
) -> str:
    """Детерминированный шаблонный текст без LLM (см. TASK_SPEC_011 §5)."""

    if not result.stationary or not result.transition_matrix:
        return '<p class="muted">Недостаточно данных для интерпретации.</p>'

    pi = result.stationary
    states = result.states
    dom_idx = pi.index(max(pi))
    dominant_label = _label(states[dom_idx].value, labels)

    next_idx = _argmax_state(result.transition_matrix[dom_idx])
    next_label = _label(states[next_idx].value, labels)
    p_next = result.transition_matrix[dom_idx][next_idx]

    tech_idx = next(
        (i for i, s in enumerate(states) if s == EpisodeState.TECHNICAL_ACTION), None
    )
    pause_idx = next(
        (i for i, s in enumerate(states) if s == EpisodeState.PAUSE), None
    )
    p_tech = pi[tech_idx] if tech_idx is not None else 0.0
    p_pause = pi[pause_idx] if pause_idx is not None else 0.0

    parts = [
        f"<p>Доминирующее состояние по стационарному распределению — "
        f"<b>{_esc(dominant_label)}</b> (π = {_fmt_float(pi[dom_idx])}).</p>",
        f"<p>Из доминирующего состояния наиболее вероятен переход в "
        f"<b>{_esc(next_label)}</b> (P = {_fmt_float(p_next)}).</p>",
        f"<p>Доля завершающих приёмов (technical_action): {_fmt_float(p_tech)}; "
        f"доля пауз: {_fmt_float(p_pause)}.</p>",
    ]
    if metrics.action_density is not None:
        parts.append(
            f"<p>Плотность действий: {_fmt_float(metrics.action_density)} действий/с.</p>"
        )
    parts.append(
        '<p class="muted">Текст сгенерирован шаблонно из числовых '
        "характеристик; никаких LLM-интерпретаций.</p>"
    )
    return "\n".join(parts)


_CSS = """
:root {
  --fg: #1f2933;
  --muted: #6b7280;
  --border: #e5e7eb;
  --accent: #1d4ed8;
  --bg: #ffffff;
  --bg-soft: #f9fafb;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
  padding: 24px;
  line-height: 1.45;
}
h1, h2, h3 { color: var(--fg); margin-top: 1.4em; }
h1 { margin-top: 0; }
.muted { color: var(--muted); font-size: 0.9em; }
.meta {
  color: var(--muted);
  font-size: 0.85em;
  margin-bottom: 16px;
}
section { margin-bottom: 28px; }
table { border-collapse: collapse; }
th, td {
  border: 1px solid var(--border);
  padding: 6px 10px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
th.row-head, th.corner, th.col-head { text-align: left; background: var(--bg-soft); }
.hm-cell { font-family: monospace; font-size: 0.85em; }
.bar-track {
  position: relative;
  width: 240px;
  height: 18px;
  background: var(--bg-soft);
  border: 1px solid var(--border);
  display: inline-block;
  vertical-align: middle;
}
.bar-fill {
  height: 100%;
  background: var(--accent);
  opacity: 0.6;
}
.bar-label {
  position: absolute;
  top: 0; left: 6px;
  font-size: 0.8em;
  color: var(--fg);
}
.warnings { padding-left: 1.2em; }
.warnings code { background: var(--bg-soft); padding: 0 4px; border-radius: 3px; }
.kv-table th { text-align: left; }
.kv-table td { text-align: left; }
.pi-table th { text-align: left; }
.pi-table td { text-align: left; }
.legend {
  display: inline-block;
  width: 12px; height: 12px;
  vertical-align: middle;
  border: 1px solid var(--border);
}
"""


def render_individual_report(
    result: MarkovIndividualResult,
    metrics: EpisodeMetrics,
    *,
    state_labels_ru: dict[str, str] | None = None,
    title: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Сформировать самодостаточный HTML-отчёт по одному спортсмену."""

    labels = state_labels_ru or _STATE_LABELS_RU
    page_title = title or f"Индивидуальная Marков-цепь — {result.athlete}"
    ts = (generated_at or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")

    body = [
        f"<h1>{_esc(page_title)}</h1>",
        f'<p class="meta">Спортсмен: <b>{_esc(result.athlete)}</b> · '
        f"режим: <code>{_esc(result.mode)}</code> · "
        f"эпизодов: {result.episode_count} · поединков: {result.bout_count} · "
        f"сгенерировано: {_esc(ts)}</p>",
        "<section><h2>Матрица переходов A</h2>",
        '<p class="muted">Строки — «откуда», столбцы — «куда». '
        "Сумма по строке = 1 ± 1e-6.</p>",
        _render_heatmap(result.transition_matrix, list(result.states), labels),
        "</section>",
        "<section><h2>Стационарное распределение π</h2>",
        _render_pi_bars(list(result.stationary), list(result.states), labels),
        "</section>",
        "<section><h2>Посещения состояний</h2>",
        _render_visit_counts(result.visit_counts, list(result.states), labels),
        "</section>",
        "<section><h2>Эпизодные метрики</h2>",
        _render_metrics(metrics),
        "</section>",
        "<section><h2>Интерпретация</h2>",
        _build_interpretation(result, metrics, labels),
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


def render_index_page(
    rendered: list[tuple[str, str]],
    *,
    title: str = "Индивидуальные Marков-модели",
    generated_at: datetime | None = None,
    summary_lines: list[str] | None = None,
) -> str:
    """Простой index с ссылками на отчёты."""

    ts = (generated_at or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
    items = "".join(
        f'<li><a href="{_esc(href)}">{_esc(name)}</a></li>'
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
        + "<section><h2>Отчёты</h2><ul class=\"reports-list\">"
        + items
        + "</ul></section>"
        "</body></html>"
    )
