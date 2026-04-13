"""Notebook-friendly MVP demo for combat analytics pipeline.

Usage in Jupyter:
    %run notebooks/mvp_demo.py
"""

from pathlib import Path
import pandas as pd

from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.pipeline import CombatHMMPipeline
from hidden_patterns_combat.preprocessing import run_preprocessing

EXCEL_PATH = Path("data/raw/episodes.xlsx")
SHEET = "Общее"
MODEL_PATH = Path("artifacts/hmm_model.pkl")
OUTPUT_DIR = Path("artifacts/analysis")

cfg = PipelineConfig()
cfg.model.n_hidden_states = 3
pipeline = CombatHMMPipeline(cfg)

pre = run_preprocessing(EXCEL_PATH, sheet_selector=SHEET, output_dir="data/processed/preprocessing")
train = pipeline.train(EXCEL_PATH, model_out=MODEL_PATH, sheet=SHEET)
analysis = pipeline.analyze(EXCEL_PATH, model_path=MODEL_PATH, output_dir=OUTPUT_DIR, sheet=SHEET)

print("Preprocessing rows:", pre.rows_cleaned)
print("Train log-likelihood:", train["log_likelihood"])
print("Analysis csv:", analysis["analysis_csv"])
print("Charts:")
for p in analysis.get("plots", []):
    print(" -", p)

episode_df = pd.read_csv(analysis["analysis_csv"])
print("\nTop 5 episodes with inferred hidden states:")
show_cols = [c for c in ["episode_id", "hidden_state", "hidden_state_name", "observed_result"] if c in episode_df.columns]
print(episode_df[show_cols].head().to_string(index=False))

summary_path = Path(analysis["summary_path"])
if summary_path.exists():
    print("\nInterpretation:\n")
    print(summary_path.read_text(encoding="utf-8"))
