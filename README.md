# hidden-patterns-combat

Проект поддерживает два режима:

1. `research` (legacy/MVP): HMM по инженерным признакам эпизода.
2. `inverse-diagnostic` (product): обратная диагностика по наблюдаемой последовательности ЗАП.

## Product постановка (inverse-diagnostic)

Скрытые состояния:
- `S1`: стойки и маневрирование
- `S2`: КФВ
- `S3`: ВУП

Наблюдаемый процесс:
- `O = {zap_r, zap_n, zap_t, hold, arm_submission, leg_submission, no_score, unknown}`

На инференсе используется `observed_zap_class` (а не инженерные признаки эпизода).

## Наблюдаемый слой: прозрачность и качество

`preprocessing/observation_builder.py` строит:
- `observed_zap_class`
- `observed_zap_source_columns`
- `observation_quality_flag`
- `observation_resolution_type`
  - `direct_finish_signal`
  - `inferred_from_score`
  - `no_score_rule`
  - `ambiguous`
  - `unknown`
- `observation_confidence_label`
  - `high`
  - `medium`
  - `low`
- `mapping_version`
- `finish_match_classes`
- `finish_match_columns`
- `score_value`, `score_rounded`, `score_supported_class`

Правила mapping остаются config-driven в:
- `src/hidden_patterns_combat/preprocessing/resources/observation_mapping_v1.json`

Score fallback сохранен, но строго вторичен:
- сначала direct finish сигналы;
- score mapping только как controlled fallback с явной маркировкой.

## Канонический слой эпизодов и последовательностей

`preprocessing/canonical_episode_table.py` строит `1 row = 1 episode` с полями:
- `sequence_id`
- `sequence_quality_flag`
- `sequence_resolution_type`
- traceability: `source_row_index`, `source_record_id`, `sheet_name`

Сегментация последовательностей:
- приоритет у явного `sequence_id/bout_id` из данных;
- иначе surrogate key (deterministic) по доступному контексту + блочной сегментации эпизодов;
- `athlete_id` не используется как единственная основа;
- низкое качество сегментации маркируется и может исключаться из train (`is_train_eligible=False`).

## Единый публичный API

И CLI, и notebook, и script используют один entrypoint:

```python
from hidden_patterns_combat.app.inverse_diagnostic_cycle import run_inverse_diagnostic_cycle
```

## Running Inverse-Diagnostic From CLI

```bash
python -m hidden_patterns_combat.cli inverse-diagnostic \
  --excel data/raw/episodes.xlsx \
  --output-dir artifacts/inverse_diagnostic \
  --n-states 3 \
  --topology-mode left_to_right
```

Эквивалентно через app entrypoint:

```bash
python -m hidden_patterns_combat.app.inverse_diagnostic_cycle \
  --input data/raw/episodes.xlsx \
  --output artifacts/inverse_diagnostic
```

## Running Inverse-Diagnostic From Notebook

Notebook frontend:
- `notebooks/inverse_diagnostic_demo.ipynb`

Он:
- вызывает `run_inverse_diagnostic_cycle(...)`;
- не дублирует бизнес-логику;
- читает и показывает артефакты inline через helpers из:
  - `hidden_patterns_combat.ui.inverse_notebook`

Основные шаги:
1. Открыть `notebooks/inverse_diagnostic_demo.ipynb`.
2. Задать `input_path`/`output_dir`/параметры.
3. Запустить ячейку pipeline.
4. Смотреть preview CSV/Markdown/plots inline.

## Артефакты inverse режима

В `output_dir` создаются:
- `cleaned/cleaned_tidy.csv`
- `cleaned/canonical_episode_table.csv`
- `cleaned/observed_sequence.csv`
- `features/hidden_state_features.csv`
- `models/inverse_hmm.pkl`
- `diagnostics/episode_analysis.csv`
- `diagnostics/state_profile.csv`
- `diagnostics/quality_diagnostics.json`
- `diagnostics/observation_audit.json`
- `diagnostics/observation_mapping_crosstab.csv`
- `diagnostics/raw_finish_signal_summary.csv`
- `diagnostics/unsupported_finish_values.csv`
- `diagnostics/unsupported_score_values.csv`
- `diagnostics/metadata_extraction_summary.json`
- `diagnostics/metadata_field_coverage.csv`
- `diagnostics/sequence_audit.json`
- `diagnostics/sequence_length_distribution.csv`
- `diagnostics/suspicious_sequences.csv`
- `diagnostics/model_health_summary.json`
- `plots/hidden_state_sequence.png`
- `plots/state_probability_profile.png`
- `reports/inverse_diagnostic_report.md`

`episode_analysis.csv` включает:
- `observed_zap_class`
- `observation_resolution_type`
- `observation_confidence_label`
- `observation_quality_flag`
- `sequence_id`
- `hidden_state`, `hidden_state_name`
- `p_state_*`, `confidence`

## Публикация аналитических результатов в репозиторий

Raw-источник остаётся локальным (`data/raw/*` не версионируется), но промежуточные и финальные аналитические артефакты можно публиковать в `analytics/runs/<run_id>/`:

```bash
PYTHONPATH=src .venv/bin/python scripts/publish_inverse_artifacts.py \
  --source artifacts/inverse_diagnostic \
  --target-root analytics/runs
```

Опционально, вместе с графиками:

```bash
PYTHONPATH=src .venv/bin/python scripts/publish_inverse_artifacts.py \
  --source artifacts/inverse_diagnostic \
  --target-root analytics/runs \
  --include-plots
```

## Backward compatibility

Старые команды сохранены:
- `train`
- `analyze`
- `preprocess`
- `demo`

Research mode (`pipeline.py`, `app/full_cycle.py`) не ломается.

## Локальная проверка

```bash
.venv/bin/pytest -q
```
