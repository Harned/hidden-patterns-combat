# hidden-patterns-combat

Исследовательский MVP для анализа скрытых закономерностей соревновательной деятельности единоборцев на основе HMM.

## Назначение модулей
- `io/` — ingestion/loading: чтение Excel, обработка multi-row headers, нормализация имен колонок.
- `preprocessing/` — базовая очистка таблиц перед feature engineering.
- `features/` — кодирование исходных бинарных признаков в компактные числовые представления.
- `modeling/` — обучение/загрузка HMM и декодирование скрытых состояний.
- `analysis/` — агрегированный профиль скрытых состояний и интерпретация.
- `visualization/` — графики последовательности скрытых состояний и вероятностного профиля.
- `reporting/` — dataclass-отчеты и генерация Markdown-отчета анализа.
- `utils/` — инфраструктурные утилиты (файлы/директории).
- `preprocessing/resources/data_dictionary_v1.json` — базовый machine-readable data dictionary.
- `pipeline.py` — базовый orchestration layer (train/analyze) между всеми слоями.
- `app/full_cycle.py` — полный orchestration layer end-to-end (ingestion -> preprocess -> features -> HMM -> analysis -> plots -> report).
- `cli.py` — пример точки входа (команды `train`, `analyze`).

Ключевые настройки MVP:
- Для основных групп бинарных признаков используется компактное кодирование `log_bitpack` (`mask=0 -> 0`, иначе `log2(mask+1)`).
- Скрытые состояния HMM по умолчанию имеют нейтральные имена `state_0`, `state_1`, ...; предметная интерпретация выводится пост-хок отдельно.

## Структура репозитория
- `src/hidden_patterns_combat/` — основной пакет.
- `data/raw/` — исходные Excel-файлы.
- `data/processed/` — промежуточные таблицы (воспроизводимость и дебаг).
- `artifacts/` — модель и аналитические результаты.
- `tests/` — unit-тесты ключевых компонентов.
- `docs/` — контекст, допущения, roadmap.
- `notebooks/` — исследовательские notebook-эксперименты.

## Быстрый старт
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Тесты:
```bash
pytest -q
```

## Точка входа (CLI)
Полный end-to-end цикл:
```bash
python scripts/run_full_cycle.py \
  --input data/raw/episodes.xlsx \
  --output artifacts/demo_run \
  --mode full
```

Или модульный запуск:
```bash
python -m hidden_patterns_combat.app.full_cycle \
  --input data/raw/episodes.xlsx \
  --output artifacts/demo_run \
  --mode fast \
  --model-path artifacts/demo_run/models/hmm_model.pkl
```

Режимы:
- `--mode full` -> `reset_outputs=True`, `retrain=True`, `save_model=True`.
- `--mode fast` -> `reset_outputs=False`, `retrain=False`, `load_existing_model=True`.

Эквивалент ручными флагами:
```bash
python scripts/run_full_cycle.py \
  --input data/raw/episodes.xlsx \
  --output artifacts/demo_run \
  --retrain \
  --reset-outputs \
  --n-states 3 \
  --random-state 42
```

Существующие команды остаются рабочими:
Пользовательский MVP-сценарий (preprocess + analyze + insight):
```bash
python -m hidden_patterns_combat.cli demo \
  --excel data/raw/episodes.xlsx \
  --sheet Общее \
  --episode-index 0 \
  --analysis-output-dir artifacts/analysis
```

Preprocessing:
```bash
python -m hidden_patterns_combat.cli preprocess \
  --excel data/raw/episodes.xlsx \
  --sheet "Общее" \
  --output-dir data/processed/preprocessing
```

Обучение:
```bash
python -m hidden_patterns_combat.cli train \
  --excel data/raw/episodes.xlsx \
  --model-out artifacts/hmm_model.pkl \
  --n-states 3
```

Анализ:
```bash
python -m hidden_patterns_combat.cli analyze \
  --excel data/raw/episodes.xlsx \
  --model artifacts/hmm_model.pkl \
  --output-dir artifacts/analysis
```

Notebook demo:
- `notebooks/mvp_ui_demo.ipynb`

## Артефакты полного цикла
`output_dir` создается со структурой:
- `cleaned/` — raw+cleaned preprocessing outputs (`raw_combined.csv`, `cleaned_tidy.csv`, `data_dictionary.csv`, `validation.json`).
- `features/` — feature sets (`raw_feature_set.csv`, `engineered_feature_set.csv`, `feature_traceability.csv`, `feature_validation.json`).
- `models/` — HMM модель (`hmm_model.pkl`, если включено сохранение).
- `diagnostics/` — `episode_analysis.csv`, `state_profile.csv`, `hmm_state_interpretation.csv`, `interpretation.txt`.
- `plots/` — графики анализа (`hidden_state_sequence.png`, `state_probability_profile.png`, и др.).
- `reports/` — итоговый `full_cycle_report.md`.

Ключевые поля итогового результата (`FullCycleResult`):
- пути к сохраненным артефактам;
- `n_rows_raw`, `n_rows_clean`, `n_sequences`, `n_features`;
- `state_summary`;
- `sample_analysis`;
- `created_artifacts`.

Назначение графиков:
- `hidden_state_sequence` — последовательность скрытых состояний по эпизодам.
- `state_probability_profile` — уверенность модели по состояниям в каждом эпизоде.
- `athlete_comparative_profile` — сравнение спортсменов по среднему результату и составу сценариев.
- `scenario_success_frequencies` — частоты успешных/неуспешных сценариев.
- `transition_distribution` — наиболее частые переходы между скрытыми состояниями.

## Notebook demo (тонкий слой)
`notebooks/mvp_ui_demo.ipynb` содержит:
1. импорт `run_full_cycle`;
2. конфиг с режимами A/B;
3. одну ячейку запуска полного цикла;
4. отображение `result.as_dict()`, sample analysis и сохраненных графиков.

Входной Excel для notebook:
- рекомендуемый путь: `data/raw/episodes.xlsx`;
- можно переопределить путь через переменную окружения:
  `HPC_INPUT_XLSX=/полный/путь/к/episodes.xlsx`;
- notebook автоматически проверяет несколько кандидатов и выводит
  понятную ошибку с перечнем проверенных путей, если файл не найден.

## Пользовательский сценарий MVP
1. Пользователь выбирает Excel-файл и (опционально) конкретный лист.
2. Система запускает preprocessing и формирует clean/raw таблицы.
3. Система строит engineered features и последовательности.
4. Система обучает модель (или переиспользует сохраненную) и выполняет анализ эпизодов.
5. Система сохраняет диагностику, графики и итоговый markdown report.
4. Пользователь получает:
   - ключевые признаки выбранного эпизода;
   - вероятный скрытый сценарий (`hidden_state_name`);
   - одну или несколько аналитических визуализаций;
   - краткий интерпретируемый текстовый вывод.

## Выходные артефакты preprocessing
- `data/processed/preprocessing/raw_combined.csv`
- `data/processed/preprocessing/cleaned_tidy.csv`
- `data/processed/preprocessing/data_dictionary.csv`
- `data/processed/preprocessing/validation.json`

Подробности словаря: `docs/data_dictionary.md`.
