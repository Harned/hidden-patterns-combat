"""Smoke-тесты для :mod:`hpc_algo.report_individual`.

Проверяем:

* HTML — самодостаточный (есть ``<style>``, нет внешних `http(s)`/`cdn`);
* содержит ключевые секции (heatmap, π-bars, метрики, warnings,
  интерпретация);
* экранирование: «инъекция» в имя спортсмена не вытекает в DOM как
  HTML-тег;
* index-страница содержит ссылки на отчёты.
"""

from __future__ import annotations

import re

from hpc_algo.episode_metrics import compute_episode_metrics
from hpc_algo.markov_individual import build_episode_sequence, fit_individual_markov
from hpc_algo.report_individual import render_index_page, render_individual_report
from hpc_algo.schema import EpisodeState
from hpc_algo.state_groups import StateGroupsConfig


def _make_cfg() -> StateGroupsConfig:
    return StateGroupsConfig(
        version="1",
        sheet="X",
        mode="single",
        priority=[
            EpisodeState.TECHNICAL_ACTION.value,
            EpisodeState.OFF_BALANCE.value,
            EpisodeState.GRIP.value,
            EpisodeState.MANOEUVRING.value,
            EpisodeState.PAUSE.value,
        ],
        states={
            "manoeuvring": ["m"],
            "grip": ["g"],
            "off_balance": ["o"],
            "technical_action": ["t"],
        },
    )


def _synthetic_result_and_metrics():
    """Маленький синтетический сценарий A→B→T для одного спортсмена."""

    from hpc_algo.episode_split import RawEpisode

    cfg = _make_cfg()
    raw_episodes = [
        RawEpisode(
            row_index=i,
            bout_id="b1",
            episode_idx_in_bout=i + 1,
            athlete="Иванов",
            episode_num_raw=float(i + 1),
            episode_time=float(10 + i),
            pause_time=2.0,
            score=None,
            feature_values={"m": 1.0 if i == 0 else 0.0,
                            "g": 1.0 if i == 1 else 0.0,
                            "o": 0.0,
                            "t": 1.0 if i == 2 else 0.0},
        )
        for i in range(3)
    ]
    records = build_episode_sequence(raw_episodes, cfg)
    result = fit_individual_markov("Иванов", records, mode="single")
    metrics = compute_episode_metrics(raw_episodes, records, athlete="Иванов")
    return result, metrics


def test_render_individual_report_contains_required_sections() -> None:
    result, metrics = _synthetic_result_and_metrics()
    html = render_individual_report(result, metrics)

    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert "</style>" in html
    # Никаких внешних ресурсов.
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn." not in html.lower()

    for marker in (
        "Матрица переходов A",
        "Стационарное распределение",
        "Эпизодные метрики",
        "Интерпретация",
        "Предупреждения модели",
        "Иванов",
    ):
        assert marker in html, f"Markup must contain marker: {marker!r}"

    # Heatmap-таблица: 5 строк состояний + 1 thead.
    assert html.count('class="hm-cell"') == 25  # 5×5

    # У каждого состояния должна быть строка-метка в π-таблице.
    assert "Маневрирование" in html
    assert "Завершающий приём" in html


def test_html_escapes_athlete_name() -> None:
    result, metrics = _synthetic_result_and_metrics()
    # Подделываем модель имени с HTML-инъекцией.
    result_dump = result.model_copy(update={"athlete": "<script>alert(1)</script>"})

    html = render_individual_report(result_dump, metrics)

    # Сырой `<script>` не должен оказаться в DOM —
    # содержимое должно быть экранировано.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_index_page_lists_athletes() -> None:
    html = render_index_page(
        [("Иванов И. И.", "ivanov.html"), ("Петров П.", "petrov.html")],
        title="Сводка",
    )
    assert html.startswith("<!doctype html>")
    assert "<title>Сводка</title>" in html
    # Обе ссылки присутствуют.
    assert re.search(r'href="ivanov\.html"', html)
    assert re.search(r'href="petrov\.html"', html)
