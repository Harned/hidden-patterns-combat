from __future__ import annotations

from hidden_patterns_combat.cli import build_parser


def test_cli_old_commands_still_parse() -> None:
    parser = build_parser()

    train = parser.parse_args(["train", "--excel", "x.xlsx", "--model-out", "m.pkl"])
    analyze = parser.parse_args(["analyze", "--excel", "x.xlsx", "--model", "m.pkl", "--output-dir", "out"])
    preprocess = parser.parse_args(["preprocess", "--excel", "x.xlsx"])
    demo = parser.parse_args(["demo", "--excel", "x.xlsx"])

    assert train.command == "train"
    assert analyze.command == "analyze"
    assert preprocess.command == "preprocess"
    assert demo.command == "demo"


def test_cli_inverse_mode_is_explicit() -> None:
    parser = build_parser()
    inverse = parser.parse_args(["inverse-diagnostic", "--excel", "x.xlsx"])
    assert inverse.command == "inverse-diagnostic"
    assert inverse.cleanup_mode is None
    assert inverse.isolated_run is False
