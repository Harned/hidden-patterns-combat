#!/usr/bin/env python3
"""Тонкая обёртка над ``hpc_algo.build_individual.build_individual_models``.

Удобна для запуска из ``Makefile`` / CI без установки console-скрипта
``hpc-algo``. Параметры по умолчанию совпадают с
``make individual-models``.

Пример::

    python scripts/build_individual_models.py \\
        --source "docs/Оценка СД содержание.xlsx" \\
        --state-groups config/state_groups.yaml \\
        --output-dir reports/individual

Никакой исследовательской логики здесь нет — только парсинг аргументов
и вызов одной функции из ``hpc_algo``. См. ``TASK_SPEC_011`` § DoD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALGO_DIR = REPO_ROOT / "algo"
if str(ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(ALGO_DIR))

from hpc_algo.build_individual import build_individual_models  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Построить ~30 индивидуальных HTML-отчётов по 5-state "
            "наблюдаемой Marков-цепи (TASK_SPEC_011)."
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
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports/individual"),
        help="Каталог HTML-отчётов и summary.json.",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Имя листа; по умолчанию берётся из YAML.",
    )
    p.add_argument(
        "--athlete",
        "-a",
        action="append",
        default=None,
        help=(
            "Точное ФИО для фильтра. Можно указывать несколько раз. "
            "Без флага — модели для всех спортсменов."
        ),
    )
    p.add_argument(
        "--style-thresholds",
        type=Path,
        default=None,
        help=(
            "YAML-пороги TASK_SPEC_013 (style_thresholds.yaml). Если задан, "
            "в HTML добавится блок «Стиль управления эпизодом»."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_individual_models(
        excel_path=args.source,
        state_groups_path=args.state_groups,
        output_dir=args.output_dir,
        sheet=args.sheet,
        athlete_filter=args.athlete,
        style_thresholds_path=args.style_thresholds,
    )
    print(
        f"OK: построено {summary.athletes_rendered} моделей "
        f"(пропущено {len(summary.skipped_athletes)}). "
        f"index: {args.output_dir}/index.html"
    )
    if summary.skipped_athletes:
        print("Пропущены без данных: " + ", ".join(summary.skipped_athletes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
