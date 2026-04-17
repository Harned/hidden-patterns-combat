#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hidden_patterns_combat.reporting.publish_artifacts import publish_inverse_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish inverse-diagnostic artifacts from local artifacts/ to versioned analytics/runs/."
    )
    parser.add_argument(
        "--source",
        default="artifacts/inverse_diagnostic",
        help="Source run directory (or parent directory containing run directories).",
    )
    parser.add_argument(
        "--target-root",
        default="analytics/runs",
        help="Versioned analytics storage root in the repository.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override run id for target folder name.",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Also publish plot PNGs from plots/*.png.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = publish_inverse_artifacts(
        source=args.source,
        target_root=args.target_root,
        run_id=args.run_id,
        include_plots=args.include_plots,
    )

    print("Published inverse-diagnostic artifacts.")
    print(f"- source_run_dir: {result.source_run_dir}")
    print(f"- target_run_dir: {result.target_run_dir}")
    print(f"- run_id: {result.run_id}")
    print(f"- copied_files: {len(result.copied_files)}")
    print(f"- missing_files: {len(result.missing_files)}")
    if result.missing_files:
        print("- missing list:")
        for item in result.missing_files:
            print(f"  - {item}")
    print(f"- publish_summary: {result.summary_path}")


if __name__ == "__main__":
    main()
