from __future__ import annotations

import argparse
import json

from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combat HMM MVP pipeline")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train HMM model")
    train.add_argument("--excel", required=True, help="Path to Excel file")
    train.add_argument("--model-out", required=True, help="Where to save model (.pkl)")
    train.add_argument("--sheet", default=None, help="Optional single sheet name")
    train.add_argument("--n-states", type=int, default=3, help="Hidden state count")

    analyze = sub.add_parser("analyze", help="Decode hidden scenario")
    analyze.add_argument("--excel", required=True, help="Path to Excel file")
    analyze.add_argument("--model", required=True, help="Path to trained model")
    analyze.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    analyze.add_argument("--sheet", default=None, help="Optional single sheet name")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Lazy import lets `--help` work even before deps are installed.
    from hidden_patterns_combat.pipeline import CombatHMMPipeline

    cfg = PipelineConfig()
    cfg.model.n_hidden_states = getattr(args, "n_states", cfg.model.n_hidden_states)
    pipeline = CombatHMMPipeline(cfg)

    if args.command == "train":
        result = pipeline.train(excel_path=args.excel, model_out=args.model_out, sheet=args.sheet)
    elif args.command == "analyze":
        result = pipeline.analyze(
            excel_path=args.excel,
            model_path=args.model,
            output_dir=args.output_dir,
            sheet=args.sheet,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
