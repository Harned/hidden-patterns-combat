from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from hidden_patterns_combat.analysis.interpreter import state_profile_table, text_summary
from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.features.encoder import encode_features, select_hmm_input_features
from hidden_patterns_combat.features.engineering import FeatureEngineeringResult, export_feature_sets
from hidden_patterns_combat.io.excel_loader import read_excel_sheets
from hidden_patterns_combat.modeling.hmm_pipeline import HMMEngine
from hidden_patterns_combat.modeling.interpretation import interpret_decoded_states
from hidden_patterns_combat.preprocessing import clean_episode_table
from hidden_patterns_combat.reporting import AnalysisReport, TrainingReport, write_analysis_markdown
from hidden_patterns_combat.utils import ensure_dir
from hidden_patterns_combat.visualization import create_analysis_charts

logger = logging.getLogger(__name__)
ParserMode = Literal["auto", "table", "matrix"]


class CombatHMMPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()

    def _load_all_rows(
        self,
        excel_path: str | Path,
        sheet: str | None = None,
        parser_mode: ParserMode = "auto",
        force_matrix_parser: bool | None = None,
    ) -> pd.DataFrame:
        sheets = read_excel_sheets(
            excel_path=excel_path,
            sheets=[sheet] if sheet else None,
            header_depth=self.cfg.header.multirow_header_depth,
            parser_mode=parser_mode,
            force_matrix_parser=force_matrix_parser,
        )
        combined = pd.concat([s.dataframe.assign(_sheet=s.name) for s in sheets], axis=0, ignore_index=True)
        combined = clean_episode_table(combined)
        logger.info("Combined dataframe shape: %s", combined.shape)
        return combined

    @staticmethod
    def _sequence_ids_from_metadata(metadata: pd.DataFrame) -> pd.Series:
        if "sequence_id" in metadata.columns:
            seq = metadata["sequence_id"].astype(str).replace({"": "sequence_0"}).fillna("sequence_0")
            return seq
        return pd.Series(["sequence_0"] * len(metadata), index=metadata.index)

    def train(
        self,
        excel_path: str | Path,
        model_out: str | Path,
        sheet: str | None = None,
        parser_mode: ParserMode = "auto",
        force_matrix_parser: bool | None = None,
    ) -> dict[str, object]:
        raw = self._load_all_rows(
            excel_path,
            sheet,
            parser_mode=parser_mode,
            force_matrix_parser=force_matrix_parser,
        )
        encoded = encode_features(raw, self.cfg.features)
        hmm_features = select_hmm_input_features(encoded.features)
        sequence_ids = self._sequence_ids_from_metadata(encoded.metadata)
        engine = HMMEngine(self.cfg.model)
        log_likelihood = engine.fit(hmm_features, sequence_ids=sequence_ids)
        engine.save(model_out)

        intermediates_dir = ensure_dir(Path(self.cfg.data_dir) / "processed")
        encoded.raw.to_csv(intermediates_dir / "training_raw_features.csv", index=False)
        encoded.features.to_csv(intermediates_dir / "training_features.csv", index=False)
        encoded.traceability.to_csv(intermediates_dir / "training_feature_traceability.csv", index=False)
        (intermediates_dir / "training_feature_validation.json").write_text(
            json.dumps(encoded.validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = TrainingReport(
            rows=len(raw),
            features=list(hmm_features.columns),
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
        parser_mode: ParserMode = "auto",
        force_matrix_parser: bool | None = None,
    ) -> dict[str, object]:
        raw = self._load_all_rows(
            excel_path,
            sheet,
            parser_mode=parser_mode,
            force_matrix_parser=force_matrix_parser,
        )
        encoded = encode_features(raw, self.cfg.features)
        hmm_features = select_hmm_input_features(encoded.features)
        sequence_ids = self._sequence_ids_from_metadata(encoded.metadata)
        engine = HMMEngine.load(model_path)
        prediction = engine.predict(hmm_features, sequence_ids=sequence_ids)

        output_dir = ensure_dir(output_dir)

        result = pd.concat(
            [
                encoded.metadata,
                encoded.features,
                pd.DataFrame(
                    {
                        "hidden_state": prediction.states,
                        "hidden_state_name": prediction.state_names,
                        "latent_state_message": [
                            f"Наиболее вероятное латентное состояние: {name}"
                            for name in prediction.state_names
                        ],
                    }
                ),
                pd.DataFrame(prediction.state_probabilities).add_prefix("p_state_"),
            ],
            axis=1,
        )
        result.to_csv(output_dir / "episode_analysis.csv", index=False)
        encoded.traceability.to_csv(output_dir / "feature_traceability.csv", index=False)
        (output_dir / "feature_validation.json").write_text(
            json.dumps(encoded.validation.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        export_feature_sets_dir = output_dir / "feature_sets"
        export_feature_sets(
            result=FeatureEngineeringResult(
                raw_feature_set=encoded.raw,
                engineered_feature_set=encoded.features,
                metadata=encoded.metadata,
                traceability=encoded.traceability,
                validation=encoded.validation,
            ),
            output_dir=export_feature_sets_dir,
        )

        state_series = pd.Series(prediction.states, name="hidden_state")
        profile = state_profile_table(encoded.features, state_series, state_definition=engine.state_definition)
        profile.to_csv(output_dir / "state_profile.csv", index=False)
        hmm_interp = interpret_decoded_states(encoded.features, state_series, engine.state_definition)
        hmm_interp.to_csv(output_dir / "hmm_state_interpretation.csv", index=False)

        summary = text_summary(profile)
        (output_dir / "interpretation.txt").write_text(summary, encoding="utf-8")

        plots: list[str] = []
        try:
            plot_map = create_analysis_charts(result, output_dir)
            plots = list(plot_map.values())
        except Exception as exc:  # pragma: no cover
            logger.warning("Plot generation skipped: %s", exc)

        report = AnalysisReport(
            rows=len(raw),
            log_likelihood=prediction.log_likelihood,
            analysis_csv=str(output_dir / "episode_analysis.csv"),
            profile_csv=str(output_dir / "state_profile.csv"),
            summary_path=str(output_dir / "interpretation.txt"),
            plots=plots,
            hidden_state_diagnostics_csv=str(output_dir / "hmm_state_interpretation.csv"),
        )
        write_analysis_markdown(report, output_dir / "report.md")
        return report.to_dict()
