from __future__ import annotations

from pathlib import Path

import pandas as pd

from hidden_patterns_combat.diagnostics.sequence_audit import build_sequence_audit, write_sequence_audit


def test_sequence_audit_flags_surrogate_and_suspicious_sequences(tmp_path: Path) -> None:
    long_rows = 60
    analysis = pd.DataFrame(
        {
            "sequence_id": (["seq_long"] * long_rows) + (["seq_short"] * 5),
            "sequence_resolution_type": (["surrogate"] * long_rows) + (["surrogate"] * 5),
            "sequence_quality_flag": (["medium"] * long_rows) + (["low"] * 5),
            "sequence_quality_reason": (["surrogate_episode_only"] * long_rows) + (["surrogate_no_context"] * 5),
            "observed_zap_class": (["no_score"] * long_rows) + ["zap_r", "zap_n", "zap_t", "no_score", "unknown"],
            "hidden_state": ([0] * long_rows) + [0, 1, 1, 2, 2],
        }
    )

    result = build_sequence_audit(analysis)

    assert result.summary["explicit_sequence_available"] is False
    assert float(result.summary["surrogate_sequence_share"]) == 1.0
    assert int(result.summary["suspicious_long_sequences"]) >= 1
    assert int(result.summary["suspicious_flat_sequences"]) >= 1
    assert any("surrogate" in warning.lower() for warning in result.summary["warnings"])

    suspicious = result.suspicious_sequences
    assert not suspicious.empty
    assert suspicious["suspicion_reasons"].astype(str).str.contains("suspiciously_long_sequence").any()

    written = write_sequence_audit(result, diagnostics_dir=tmp_path)
    assert Path(written["sequence_audit_json"]).exists()
    assert Path(written["sequence_length_distribution_csv"]).exists()
    assert Path(written["suspicious_sequences_csv"]).exists()
