from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from hidden_patterns_combat.analysis.interpreter import state_profile_table, text_summary
from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.features.encoder import encode_features
from hidden_patterns_combat.io.excel_loader import read_excel_sheets
from hidden_patterns_combat.modeling.hmm_pipeline import HMMEngine
from hidden_patterns_combat.preprocessing import clean_episode_table
from hidden_patterns_combat.reporting import AnalysisReport, TrainingReport, write_analysis_markdown
from hidden_patterns_combat.utils import ensure_dir

logger = logging.getLogger(__name__)


class CombatHMMPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()

    def _load_all_rows(self, excel_path: str | Path, sheet: str | None = None) -> pd.DataFrame:
        sheets = read_excel_sheets(
            excel_path=excel_path,
            sheets=[sheet] if sheet else None,
            header_depth=self.cfg.header.multirow_header_depth,
        )
        combined = pd.concat([s.dataframe.assign(_sheet=s.name) for s in sheets], axis=0, ignore_index=True)
        combined = clean_episode_table(combined)
        logger.info("Combined dataframe shape: %s", combined.shape)
        return combined

    def train(self, excel_path: str | Path, model_out: str | Path, sheet: str | None = None) -> dict[str, object]:
        raw = self._load_all_rows(excel_path, sheet)
        encoded = encode_features(raw, self.cfg.features)
        engine = HMMEngine(self.cfg.model)
        log_likelihood = engine.fit(encoded.features)
        engine.save(model_out)

        intermediates_dir = ensure_dir(Path(self.cfg.data_dir) / "processed")
        encoded.features.to_csv(intermediates_dir / "training_features.csv", index=False)

        report = TrainingReport(
            rows=len(raw),
            features=list(encoded.features.columns),
            log_likelihood=log_likelihood,
            model_path=str(model_out),
        )
        return report.to_dict()

    def analyze(
        self,
        excel_path: str | Path,
        model_path: str | Path,
        output_dir: str | Path,
        sheet: str | None = None,
    ) -> dict[str, object]:
        raw = self._load_all_rows(excel_path, sheet)
        encoded = encode_features(raw, self.cfg.features)
        engine = HMMEngine.load(model_path)
        prediction = engine.predict(encoded.features)

        output_dir = ensure_dir(output_dir)

        result = pd.concat(
            [
                encoded.metadata,
                encoded.features,
                pd.DataFrame({"hidden_state": prediction.states}),
                pd.DataFrame(prediction.state_probabilities).add_prefix("p_state_"),
            ],
            axis=1,
        )
        result.to_csv(output_dir / "episode_analysis.csv", index=False)

        state_series = pd.Series(prediction.states, name="hidden_state")
        profile = state_profile_table(encoded.features, state_series)
        profile.to_csv(output_dir / "state_profile.csv", index=False)

        summary = text_summary(profile)
        (output_dir / "interpretation.txt").write_text(summary, encoding="utf-8")

        plots: list[str] = []
        try:
            from hidden_patterns_combat.visualization.plots import (
                plot_hidden_states,
                plot_state_probabilities,
            )

            hidden_states_path = output_dir / "hidden_states.png"
            state_probs_path = output_dir / "state_probabilities.png"
            plot_hidden_states(state_series, hidden_states_path)
            plot_state_probabilities(prediction.state_probabilities, state_probs_path)
            plots = [str(hidden_states_path), str(state_probs_path)]
        except Exception as exc:  # pragma: no cover
            logger.warning("Plot generation skipped: %s", exc)

        report = AnalysisReport(
            rows=len(raw),
            log_likelihood=prediction.log_likelihood,
            analysis_csv=str(output_dir / "episode_analysis.csv"),
            profile_csv=str(output_dir / "state_profile.csv"),
            summary_path=str(output_dir / "interpretation.txt"),
            plots=plots,
        )
        write_analysis_markdown(report, output_dir / "report.md")
        return report.to_dict()
