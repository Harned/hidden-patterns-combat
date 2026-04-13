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
    train.add_argument("--force-matrix-parser", action="store_true", help="Force matrix-style parser for loading.")

    analyze = sub.add_parser("analyze", help="Decode hidden scenario")
    analyze.add_argument("--excel", required=True, help="Path to Excel file")
    analyze.add_argument("--model", required=True, help="Path to trained model")
    analyze.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    analyze.add_argument("--sheet", default=None, help="Optional single sheet name")
    analyze.add_argument("--force-matrix-parser", action="store_true", help="Force matrix-style parser for loading.")

    preprocess = sub.add_parser("preprocess", help="Preprocess Excel into tidy tabular format")
    preprocess.add_argument("--excel", required=True, help="Path to Excel file")
    preprocess.add_argument(
        "--sheet",
        action="append",
        default=None,
        help="Sheet name (repeat flag for multiple sheets). Omit to process all sheets.",
    )
    preprocess.add_argument("--header-depth", type=int, default=2, help="Excel header depth")
    preprocess.add_argument(
        "--force-matrix-parser",
        action="store_true",
        help="Force matrix-style parser for selected sheets.",
    )
    preprocess.add_argument(
        "--output-dir",
        default="data/processed/preprocessing",
        help="Directory for raw/cleaned exports",
    )
    preprocess.add_argument("--save-parquet", action="store_true", help="Also export parquet files")

    demo = sub.add_parser("demo", help="Run end-user MVP workflow (preprocess + analyze + insight)")
    demo.add_argument("--excel", required=True, help="Path to Excel file")
    demo.add_argument("--sheet", default=None, help="Optional single sheet name")
    demo.add_argument("--model", default="artifacts/hmm_model.pkl", help="Path to model file")
    demo.add_argument("--n-states", type=int, default=3, help="Hidden state count for training")
    demo.add_argument("--episode-index", type=int, default=None, help="Episode index to inspect")
    demo.add_argument("--retrain", action="store_true", help="Retrain model before analysis")
    demo.add_argument("--force-matrix-parser", action="store_true", help="Force matrix-style parser for loading.")
    demo.add_argument(
        "--preprocess-output-dir",
        default="data/processed/preprocessing",
        help="Directory for preprocessing artifacts",
    )
    demo.add_argument(
        "--analysis-output-dir",
        default="artifacts/analysis",
        help="Directory for analysis artifacts",
    )

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
        result = pipeline.train(
            excel_path=args.excel,
            model_out=args.model_out,
            sheet=args.sheet,
            force_matrix_parser=args.force_matrix_parser,
        )
    elif args.command == "analyze":
        result = pipeline.analyze(
            excel_path=args.excel,
            model_path=args.model,
            output_dir=args.output_dir,
            sheet=args.sheet,
            force_matrix_parser=args.force_matrix_parser,
        )
    elif args.command == "preprocess":
        from hidden_patterns_combat.preprocessing import run_preprocessing

        sheet_selector: str | list[str] | None
        if args.sheet is None:
            sheet_selector = None
        elif len(args.sheet) == 1:
            sheet_selector = args.sheet[0]
        else:
            sheet_selector = args.sheet

        result = run_preprocessing(
            excel_path=args.excel,
            sheet_selector=sheet_selector,
            header_depth=args.header_depth,
            output_dir=args.output_dir,
            save_parquet=args.save_parquet,
            force_matrix_parser=args.force_matrix_parser,
        ).to_dict()
    elif args.command == "demo":
        from hidden_patterns_combat.ui import run_demo_workflow

        result = run_demo_workflow(
            excel_path=args.excel,
            sheet=args.sheet,
            model_path=args.model,
            preprocess_output_dir=args.preprocess_output_dir,
            analysis_output_dir=args.analysis_output_dir,
            episode_index=args.episode_index,
            n_states=args.n_states,
            retrain=args.retrain,
            force_matrix_parser=args.force_matrix_parser,
        ).to_dict()
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
