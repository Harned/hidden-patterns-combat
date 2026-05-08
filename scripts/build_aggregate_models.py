#!/usr/bin/env python3
"""Тонкая обёртка над ``hpc_algo.build_aggregate.build_aggregate_models``.

Для запуска из ``Makefile`` / CI без установки console-script ``hpc-algo``.
Параметры по умолчанию совпадают с ``make aggregate-models``.

Пример::

    python scripts/build_aggregate_models.py \\
        --source "docs/Оценка СД содержание.xlsx" \\
        --state-groups config/state_groups.yaml \\
        --finalists config/finalists.yaml \\
        --output-dir reports/aggregate \\
        --alpha 0.5

Никакой исследовательской логики; только парсинг аргументов и вызов
функции ядра.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALGO_DIR = REPO_ROOT / "algo"
if str(ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(ALGO_DIR))

from hpc_algo.build_aggregate import build_aggregate_models  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Построить агрегатные 5-state Marков-модели по призёрам 1–3 "
            "каждой весовой категории (TASK_SPEC_012)."
        ),
    )
    p.add_argument(
        "--source",
        "-s",
        type=Path,
        default=Path("docs/Оценка СД содержание.xlsx"),
        help="Excel-источник.",
    )
    p.add_argument(
        "--state-groups",
        "-c",
        type=Path,
        default=Path("config/state_groups.yaml"),
        help="YAML-конфиг 5-state алфавита.",
    )
    p.add_argument(
        "--finalists",
        "-f",
        type=Path,
        default=None,
        help=(
            "YAML со списком призёров {weight_class: {place: athlete}}. "
            "Если не задан — fallback на колонки Excel."
        ),
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports/aggregate"),
        help="Каталог HTML-отчётов и summary.json.",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Имя листа; по умолчанию берётся из state-groups YAML.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Вес L1 в composite-метрике; default=0.5.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_aggregate_models(
        excel_path=args.source,
        state_groups_path=args.state_groups,
        output_dir=args.output_dir,
        finalists_path=args.finalists,
        sheet=args.sheet,
        alpha=args.alpha,
    )
    print(
        f"OK: построено {summary.weight_classes_rendered} агрегатов "
        f"(пропущено {len(summary.skipped_weight_classes)}). "
        f"index: {args.output_dir}/index.html"
    )
    if summary.skipped_weight_classes:
        print("Пропущенные категории: " + ", ".join(summary.skipped_weight_classes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
