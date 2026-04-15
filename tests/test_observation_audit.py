from __future__ import annotations

from pathlib import Path

import pandas as pd

from hidden_patterns_combat.diagnostics.observation_audit import (
    build_observation_audit,
    write_observation_audit,
)
from hidden_patterns_combat.preprocessing.observation_builder import (
    build_observed_zap_classes,
    load_observation_mapping_config,
)


def test_observation_audit_detects_direct_vs_score_and_unsupported_signals(tmp_path: Path) -> None:
    cleaned = pd.DataFrame(
        {
            "outcomes__score": [1, 2, 0, 5, None, 1],
            "outcomes__finish_action_04_05": [1, 0, 0, 0, 0, 0],  # hold
            "outcomes__finish_action_05_06": [0, 1, 0, 0, 0, 0],  # arm_submission
            "custom_finish_signal": [0, 1, 0, 0, 0, 0],  # unsupported raw finish-like column
        }
    )
    cfg = load_observation_mapping_config()
    obs_result = build_observed_zap_classes(cleaned, config=cfg)

    audit = build_observation_audit(
        cleaned_df=cleaned,
        observation_df=obs_result.observations,
        cfg=cfg,
        score_column=obs_result.score_column,
        finish_signal_columns=obs_result.finish_signal_columns,
    )

    assert audit.summary["direct_finish_signal_columns_available"] is True
    assert float(audit.summary["direct_finish_positive_share"]) > 0.0
    assert 5 in audit.summary["unsupported_score_values"]
    assert "custom_finish_signal" in audit.summary["unsupported_finish_columns_with_positive_values"]

    resolution_share = audit.summary.get("resolution_type_share", {})
    assert float(resolution_share.get("direct_finish_signal", 0.0)) > 0.0
    assert float(resolution_share.get("inferred_from_score", 0.0)) > 0.0

    written = write_observation_audit(audit, diagnostics_dir=tmp_path)
    for key in (
        "observation_audit_json",
        "observation_mapping_crosstab_csv",
        "raw_finish_signal_summary_csv",
        "unsupported_score_values_csv",
    ):
        assert Path(written[key]).exists()
