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


if __name__ == "__main__":
    app()
