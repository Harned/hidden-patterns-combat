from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hidden_patterns_combat.app.full_cycle import FullCycleResult, run_full_cycle


HAS_HMMLEARN = importlib.util.find_spec("hmmlearn") is not None


def test_output_structure_created_even_if_model_missing(demo_excel_path: Path, tmp_path: Path):
    output_dir = tmp_path / "full_cycle_outputs"

    with pytest.raises(FileNotFoundError):
        run_full_cycle(
            input_path=demo_excel_path,
            output_dir=output_dir,
            retrain=False,
            load_existing_model=True,
            model_path=output_dir / "models" / "missing.pkl",
            generate_plots=False,
            verbose=False,
        )

    for name in ["cleaned", "features", "models", "plots", "reports", "diagnostics"]:
        assert (output_dir / name).exists()


def test_reset_outputs_removes_old_files(demo_excel_path: Path, tmp_path: Path):
    output_dir = tmp_path / "full_cycle_outputs"
    stale_file = output_dir / "cleaned" / "old.txt"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("stale", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        run_full_cycle(
            input_path=demo_excel_path,
            output_dir=output_dir,
            retrain=False,
            load_existing_model=True,
            model_path=output_dir / "models" / "missing.pkl",
            reset_outputs=True,
            generate_plots=False,
            verbose=False,
        )

    assert not stale_file.exists()


def test_reuse_mode_requires_existing_model(demo_excel_path: Path, tmp_path: Path):
    missing_model = tmp_path / "no_model.pkl"

    with pytest.raises(FileNotFoundError):
        run_full_cycle(
            input_path=demo_excel_path,
            output_dir=tmp_path / "artifacts",
            retrain=False,
            load_existing_model=True,
            model_path=missing_model,
            generate_plots=False,
            verbose=False,
        )


@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn missing")
def test_full_cycle_returns_result_dataclass(demo_excel_path: Path, tmp_path: Path):
    output_dir = tmp_path / "run_train"

    result = run_full_cycle(
        input_path=demo_excel_path,
        output_dir=output_dir,
        retrain=True,
        save_model=True,
        generate_plots=False,
        verbose=False,
    )

    assert isinstance(result, FullCycleResult)
    assert result.n_rows_raw >= result.n_rows_clean >= 1
    assert result.n_features >= 1
    assert result.n_sequences >= 1
    assert result.model_path is not None
    assert Path(result.model_path).exists()
    assert Path(result.cleaned_data_path).exists()
    assert Path(result.features_path).exists()
    assert Path(result.report_path).exists()


@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn missing")
def test_full_cycle_fast_reuse_mode_smoke(demo_excel_path: Path, tmp_path: Path):
    output_dir = tmp_path / "run_reuse"

    first = run_full_cycle(
        input_path=demo_excel_path,
        output_dir=output_dir,
        retrain=True,
        save_model=True,
        generate_plots=False,
        verbose=False,
    )
    assert first.model_path is not None

    second = run_full_cycle(
        input_path=demo_excel_path,
        output_dir=output_dir,
        retrain=False,
        load_existing_model=True,
        model_path=first.model_path,
        generate_plots=False,
        verbose=False,
    )

    payload = second.as_dict()
    assert payload["n_rows_clean"] >= 1
    assert payload["state_summary"]
    assert payload["sample_analysis"]
    assert str(payload["sample_analysis"].get("episode_id", "")).lower() not in {"", "nan", "none"}
    assert str(payload["sample_analysis"].get("sequence_id", "")).lower() not in {"", "nan", "none"}
    assert isinstance(payload.get("canonical_state_mapping"), dict)
    assert isinstance(payload.get("observed_signal"), dict)
    assert payload["observed_signal"].get("classification") in {"proxy", "direct_zap"}
    assert isinstance(payload.get("consistency_warnings"), list)
    assert Path(payload["diagnostics_path"]).exists()


@pytest.mark.skipif(not HAS_HMMLEARN, reason="hmmlearn missing")
def test_full_cycle_respects_explicit_sheet_selection(demo_excel_path: Path, tmp_path: Path):
    output_dir = tmp_path / "run_sheet_selection"
    result = run_full_cycle(
        input_path=demo_excel_path,
        output_dir=output_dir,
        sheet_names=["Общее"],
        n_states=2,
        retrain=True,
        save_model=False,
        generate_plots=False,
        verbose=False,
    )
    assert result.requested_sheets == ["Общее"]
    assert result.loaded_sheets == ["Общее"]
