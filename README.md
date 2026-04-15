# hidden-patterns-combat

Проект теперь поддерживает **два явных режима**:

1. `research` (обратная совместимость MVP): HMM по инженерным признакам эпизода.
2. `inverse-diagnostic` (продуктовый режим): обратная задача HMM по наблюдаемой последовательности `observed_zap_class`.

## Формальная постановка

Скрытый процесс (латентные состояния):
- `S1` = стойки и маневрирование
- `S2` = КФВ
- `S3` = ВУП

Наблюдаемый процесс (inverse mode):
- `O = {zap_r, zap_n, zap_t, hold, arm_submission, leg_submission, no_score, unknown}`

Важное разделение:
- В обучении inverse режима используются исторические данные для оценки параметров.
- В инференсе по новой схватке используется только наблюдаемая последовательность `O_t`.
- Hidden-state feature layer используется для semantic init/post-hoc relabeling, но не как shortcut наблюдений.

## Архитектура

- `preprocessing/observation_builder.py`:
  - строит `observed_zap_class`
  - добавляет `observed_zap_source_columns`, `observation_quality_flag`, `mapping_version`
  - правила вынесены в `preprocessing/resources/observation_mapping_v1.json`
- `preprocessing/canonical_episode_table.py`:
  - нормализованная таблица `1 строка = 1 эпизод`
  - traceability: `source_row_index`, `source_record_id`, `sheet_name`
  - фильтрация total rows, `is_train_eligible`
- `features/hidden_state_features.py`:
  - явный слой скрытых признаков (`hidden_state_features`) отдельно от наблюдаемой последовательности
- `modeling/inverse_hmm.py`:
  - discrete HMM для inverse задачи (Viterbi + posterior)
  - weak supervision/semantic initialization и post-hoc relabeling S1/S2/S3
- `app/inverse_diagnostic_cycle.py`:
  - end-to-end продуктовый inverse pipeline и отчёт `inverse_diagnostic_report.md`

Старый контур сохранен:
- `pipeline.py` и `app/full_cycle.py` продолжают работать без изменения контрактов.

## CLI

### Старые команды (совместимость)

```bash
python -m hidden_patterns_combat.cli train --excel data/raw/episodes.xlsx --model-out artifacts/hmm_model.pkl
python -m hidden_patterns_combat.cli analyze --excel data/raw/episodes.xlsx --model artifacts/hmm_model.pkl --output-dir artifacts/analysis
python -m hidden_patterns_combat.cli preprocess --excel data/raw/episodes.xlsx --output-dir data/processed/preprocessing
python -m hidden_patterns_combat.cli demo --excel data/raw/episodes.xlsx
```

### Новый продуктовый inverse режим

```bash
python -m hidden_patterns_combat.cli inverse-diagnostic \
  --excel data/raw/episodes.xlsx \
  --output-dir artifacts/inverse_diagnostic \
  --n-states 3 \
  --topology-mode left_to_right
```

Эквивалент через app entrypoint:

```bash
python -m hidden_patterns_combat.app.inverse_diagnostic_cycle \
  --input data/raw/episodes.xlsx \
  --output artifacts/inverse_diagnostic
```

## Артефакты inverse режима

В `output_dir` создаются:
- `cleaned/cleaned_tidy.csv`
- `cleaned/canonical_episode_table.csv`
- `cleaned/observed_sequence.csv`
- `features/hidden_state_features.csv`
- `models/inverse_hmm.pkl`
- `diagnostics/episode_analysis.csv`
- `diagnostics/state_profile.csv`
- `plots/hidden_state_sequence.png`
- `plots/state_probability_profile.png`
- `reports/inverse_diagnostic_report.md`

В `episode_analysis.csv` присутствуют:
- `observed_zap_class`
- `hidden_state`, `hidden_state_name`
- `p_state_*`
- `confidence`
- `observation_quality_flag`

## Ограничения продукта

- `unknown` означает недостаточность/неоднозначность данных маппинга, а не отдельный тактический приём.
- `no_score` — явный класс для нерезультативных эпизодов (score=0 и нет завершающего действия).
- Модель не делает причинных выводов; это диагностический вероятностный слой.
- Рекомендация может быть помечена как недостаточно уверенная при низком posterior/высокой доле `unknown`.

## Локальная проверка

```bash
.venv/bin/pytest -q
```

