from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TrainingReport:
    rows: int
    features: list[str]
    log_likelihood: float
    model_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AnalysisReport:
    rows: int
    log_likelihood: float
    analysis_csv: str
    profile_csv: str
    summary_path: str
    plots: list[str]
    hidden_state_diagnostics_csv: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
