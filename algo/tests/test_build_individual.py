"""End-to-end тесты оркестратора :mod:`hpc_algo.build_individual`.

Используем фикстуру ``markov_two_bouts_excel`` (3-уровневая шапка,
2 поединка, ep_num reset, значение ``8`` в признаковой колонке).
Генерируем YAML на лету по реальным flatten-именам колонок и
запускаем полный пайплайн — без сетевых ресурсов, без HMM.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from hpc_algo.build_individual import _slugify, build_individual_models
from hpc_algo.episode_split import read_episodes_sheet


def _yaml_for_fixture(excel_path: Path, yaml_path: Path) -> dict[str, list[str]]:
    """Сгенерировать config/state_groups.yaml под колонки фикстуры."""

    df = read_episodes_sheet(excel_path, sheet="Общее")
    cols = list(df.columns)

    states = {
        "manoeuvring": [c for c in cols if "Стойка" in c],
        "grip": [c for c in cols if "КФВ" in c],
        "off_balance": [c for c in cols if "ВУП" in c],
        "technical_action": [
            c for c in cols if ("Завершающие" in c or "Болевой" in c)
        ],
    }
    payload = {
        "version": "1",
        "sheet": "Общее",
        "mode": "single",
        "priority": [
            "technical_action",
            "off_balance",
            "grip",
            "manoeuvring",
            "pause",
        ],
        "states": states,
    }
    yaml_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return states


def test_build_individual_models_end_to_end(
    markov_two_bouts_excel: Path, tmp_path: Path
) -> None:
    yaml_path = tmp_path / "state_groups.yaml"
    _yaml_for_fixture(markov_two_bouts_excel, yaml_path)
    out_dir = tmp_path / "reports_individual"

    summary = build_individual_models(
        excel_path=markov_two_bouts_excel,
        state_groups_path=yaml_path,
        output_dir=out_dir,
        sheet="Общее",
    )

    assert summary.athletes_total >= 2  # Иванов, Петров
    assert summary.athletes_rendered == summary.athletes_total
    assert set(summary.rendered_athletes) >= {"Иванов", "Петров"}
    assert (out_dir / "index.html").exists()
    assert (out_dir / "summary.json").exists()

    # На каждого спортсмена есть HTML-файл.
    for athlete in summary.rendered_athletes:
        html_path = out_dir / f"{_slugify(athlete)}.html"
        assert html_path.exists(), f"Missing HTML for {athlete}"
        text = html_path.read_text(encoding="utf-8")
        assert "<!doctype html>" in text.lower()
        assert "Матрица переходов" in text

    # >2 → агрегированный warning попал в split_warnings.
    codes = {w.code for w in summary.split_warnings}
    assert "data.value_out_of_range" in codes


def test_build_individual_models_athlete_filter(
    markov_two_bouts_excel: Path, tmp_path: Path
) -> None:
    yaml_path = tmp_path / "state_groups.yaml"
    _yaml_for_fixture(markov_two_bouts_excel, yaml_path)
    out_dir = tmp_path / "reports_filtered"

    summary = build_individual_models(
        excel_path=markov_two_bouts_excel,
        state_groups_path=yaml_path,
        output_dir=out_dir,
        sheet="Общее",
        athlete_filter=["Иванов"],
    )

    assert summary.rendered_athletes == ["Иванов"]
    assert (out_dir / f"{_slugify('Иванов')}.html").exists()
    assert not (out_dir / f"{_slugify('Петров')}.html").exists()


def test_build_individual_models_summary_json_is_valid(
    markov_two_bouts_excel: Path, tmp_path: Path
) -> None:
    yaml_path = tmp_path / "state_groups.yaml"
    _yaml_for_fixture(markov_two_bouts_excel, yaml_path)
    out_dir = tmp_path / "reports_summary"

    summary = build_individual_models(
        excel_path=markov_two_bouts_excel,
        state_groups_path=yaml_path,
        output_dir=out_dir,
        sheet="Общее",
    )

    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["athletes_rendered"] == summary.athletes_rendered
    assert payload["sheet"] == "Общее"
    # warnings — сериализуемы; счётчики корректны.
    assert isinstance(payload["per_athlete_warning_counts"], dict)


def test_slugify_handles_cyrillic_and_punctuation() -> None:
    assert _slugify("Иванов И. И.") == "Иванов_И_И"
    assert _slugify("  ") == "athlete"
    assert _slugify("Petrov, P.") == "Petrov_P"


def test_build_individual_models_with_style_thresholds(
    markov_two_bouts_excel: Path, tmp_path: Path
) -> None:
    """С style_thresholds.yaml HTML включает блок стиля.

    Подбираем порог так, чтобы хотя бы один спортсмен попал в
    `speed_power`. Проверяем, что блок отрисовался и имя стиля попало в
    HTML; конкретный label не фиксируем, чтобы тест не зависел от
    точного содержимого фикстуры.
    """

    yaml_path = tmp_path / "state_groups.yaml"
    _yaml_for_fixture(markov_two_bouts_excel, yaml_path)
    out_dir = tmp_path / "reports_styled"

    style_path = tmp_path / "style.yaml"
    style_path.write_text(
        """
thresholds:
  endurance:
    min_episode_count: 3
    min_activity_evenness: 0.0
  speed_power:
    max_episode_count: 100
    min_action_density: 0.0
  burnout:
    min_action_density_first_half: 100.0
    max_action_density_second_half: 0.0
""",
        encoding="utf-8",
    )

    summary = build_individual_models(
        excel_path=markov_two_bouts_excel,
        state_groups_path=yaml_path,
        output_dir=out_dir,
        sheet="Общее",
        style_thresholds_path=style_path,
    )
    assert summary.athletes_rendered >= 1

    # Проверяем хотя бы один HTML.
    sample = out_dir / f"{_slugify(summary.rendered_athletes[0])}.html"
    text = sample.read_text(encoding="utf-8")
    assert "Стиль управления эпизодом" in text
    # Один из четырёх известных лейблов (включая unclassified) обязан
    # присутствовать как код.
    assert any(
        f"<code>{label}</code>" in text
        for label in ("endurance", "speed_power", "burnout", "unclassified")
    )
