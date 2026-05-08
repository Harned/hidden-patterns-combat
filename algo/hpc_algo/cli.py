"""CLI для независимого запуска processing module.

Примеры::

    hpc-algo analyze docs/"Оценка СД содержание.xlsx"
    hpc-algo analyze path.xlsx --output result.json
    hpc-algo summary path.xlsx

CLI не содержит исследовательской логики: это только тонкая обёртка над
:func:`hpc_algo.api.analyze_source`.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hpc_algo.api import AnalyzeConfig, analysis_summary, analyze_source, preflight_mapping

app = typer.Typer(
    help="Independent processing module for hidden-patterns-combat.",
    add_completion=False,
    no_args_is_help=True,
)


def _dump(data: object, output: Path | None) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if output is None:
        typer.echo(payload)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"OK: результат сохранён в {output}")


@app.command("analyze")
def analyze_cmd(
    source: Path = typer.Argument(..., exists=True, readable=True, help="Путь к Excel-источнику."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Путь для сохранения AnalysisResult в JSON. По умолчанию — stdout.",
    ),
    summary_only: bool = typer.Option(
        False,
        "--summary",
        help="Показать только компактную сводку, без audit/charts.",
    ),
) -> None:
    """Выполнить honest-анализ Excel-источника."""

    result = analyze_source(source)

    if summary_only:
        _dump(analysis_summary(result), output)
        return

    # pydantic v2 -> json-совместимый dict.
    payload = json.loads(result.model_dump_json())
    _dump(payload, output)


@app.command("summary")
def summary_cmd(
    source: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Компактная сводка (status + detected + warnings) по источнику."""

    result = analyze_source(source)
    _dump(analysis_summary(result), output)


@app.command("preflight")
def preflight_cmd(
    source: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Предложить ColumnMappingConfig на основе эвристики."""

    cfg = preflight_mapping(source)
    _dump(json.loads(cfg.model_dump_json()), output)


@app.command("analyze-with-mapping")
def analyze_with_mapping_cmd(
    source: Path = typer.Argument(..., exists=True, readable=True),
    mapping_path: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Выполнить анализ с переданным файлом column mapping (JSON)."""

    from hpc_algo.schema import ColumnMappingConfig

    cfg = ColumnMappingConfig.model_validate_json(
        mapping_path.read_text(encoding="utf-8")
    )
    result = analyze_source(source, AnalyzeConfig(column_mapping=cfg))
    _dump(json.loads(result.model_dump_json()), output)


@app.command("report")
def report_cmd(
    source: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Короткий текстовый отчёт (ru)."""

    result = analyze_source(source)
    typer.echo(result.report)
    typer.echo("")
    if result.warnings:
        typer.echo("Предупреждения:")
        for w in result.warnings:
            typer.echo(f"  [{w.severity.value}] {w.code}: {w.message}")
    if result.errors:
        typer.echo("")
        typer.echo("Ошибки:")
        for e in result.errors:
            typer.echo(f"  [{e.severity.value}] {e.code}: {e.message}")


@app.command("individual-markov")
def individual_markov_cmd(
    source: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Excel-источник (например, docs/Оценка СД содержание.xlsx).",
    ),
    state_groups: Path = typer.Option(
        Path("config/state_groups.yaml"),
        "--state-groups",
        "-c",
        exists=True,
        readable=True,
        help="YAML-конфиг 5-state алфавита.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/individual"),
        "--output-dir",
        "-o",
        help="Каталог для HTML-отчётов и summary.json.",
    ),
    sheet: str | None = typer.Option(
        None,
        "--sheet",
        help="Имя листа; по умолчанию берётся из YAML (`sheet`, default `Общее`).",
    ),
    athlete: list[str] = typer.Option(
        None,
        "--athlete",
        "-a",
        help=(
            "Точное имя ФИО для фильтра. Можно повторять флаг несколько раз."
            " Без флага — модели для всех найденных спортсменов."
        ),
    ),
    style_thresholds: Path | None = typer.Option(
        None,
        "--style-thresholds",
        help=(
            "YAML-файл порогов TASK_SPEC_013. Если задан — в HTML добавится"
            " блок «Стиль управления эпизодом» (endurance / speed_power /"
            " burnout / unclassified)."
        ),
    ),
) -> None:
    """Построить ~30 индивидуальных Marков-моделей по эпизодам (Уровень 1)."""

    from hpc_algo.build_individual import build_individual_models

    summary = build_individual_models(
        excel_path=source,
        state_groups_path=state_groups,
        output_dir=output_dir,
        sheet=sheet,
        athlete_filter=athlete or None,
        style_thresholds_path=style_thresholds,
    )
    typer.echo(
        f"OK: построено {summary.athletes_rendered} моделей "
        f"(пропущено {len(summary.skipped_athletes)}). "
        f"index: {output_dir}/index.html"
    )
    if summary.skipped_athletes:
        typer.echo(
            "Пропущены без данных: " + ", ".join(summary.skipped_athletes)
        )


@app.command("aggregate-markov")
def aggregate_markov_cmd(
    source: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Excel-источник (например, docs/Оценка СД содержание.xlsx).",
    ),
    state_groups: Path = typer.Option(
        Path("config/state_groups.yaml"),
        "--state-groups",
        "-c",
        exists=True,
        readable=True,
        help="YAML-конфиг 5-state алфавита (тот же, что в individual-markov).",
    ),
    finalists: Path | None = typer.Option(
        None,
        "--finalists",
        "-f",
        help=(
            "YAML со списком призёров {weight_class: {place: athlete}}."
            " Если не задан — пробуем извлечь из колонок Excel"
            " ('Весовая категория' / 'Место')."
        ),
    ),
    output_dir: Path = typer.Option(
        Path("reports/aggregate"),
        "--output-dir",
        "-o",
        help="Каталог HTML-отчётов и summary.json.",
    ),
    sheet: str | None = typer.Option(
        None,
        "--sheet",
        help="Имя листа; по умолчанию берётся из YAML state-groups.",
    ),
    alpha: float = typer.Option(
        0.5,
        "--alpha",
        min=0.0,
        max=1.0,
        help=(
            "Вес L1 в composite-метрике; default=0.5. composite ="
            " α·L1(π) + (1-α)·KL(A)."
        ),
    ),
) -> None:
    """Построить агрегатные модели по призёрам 1–3 каждой весовой категории."""

    from hpc_algo.build_aggregate import build_aggregate_models

    summary = build_aggregate_models(
        excel_path=source,
        state_groups_path=state_groups,
        output_dir=output_dir,
        finalists_path=finalists,
        sheet=sheet,
        alpha=alpha,
    )
    typer.echo(
        f"OK: построено {summary.weight_classes_rendered} агрегатов "
        f"(пропущено {len(summary.skipped_weight_classes)}). "
        f"index: {output_dir}/index.html"
    )
    if summary.skipped_weight_classes:
        typer.echo("Пропущенные категории: " + ", ".join(summary.skipped_weight_classes))


if __name__ == "__main__":
    app()
