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
- `pipeline.py` — orchestration layer (train/analyze) между всеми слоями.
- `cli.py` — пример точки входа (команды `train`, `analyze`).

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

## Выходные артефакты анализа
- `artifacts/analysis/episode_analysis.csv`
- `artifacts/analysis/state_profile.csv`
- `artifacts/analysis/hmm_state_interpretation.csv`
- `artifacts/analysis/interpretation.txt`
- `artifacts/analysis/report.md`
- `artifacts/analysis/hidden_state_sequence.png`
- `artifacts/analysis/state_probability_profile.png`
- `artifacts/analysis/athlete_comparative_profile.png` (если в данных есть `athlete_name`)
- `artifacts/analysis/scenario_success_frequencies.png`
- `artifacts/analysis/transition_distribution.png`

Назначение графиков:
- `hidden_state_sequence` — последовательность скрытых состояний по эпизодам.
- `state_probability_profile` — уверенность модели по состояниям в каждом эпизоде.
- `athlete_comparative_profile` — сравнение спортсменов по среднему результату и составу сценариев.
- `scenario_success_frequencies` — частоты успешных/неуспешных сценариев.
- `transition_distribution` — наиболее частые переходы между скрытыми состояниями.

## Пользовательский сценарий MVP
1. Пользователь выбирает Excel-файл и (опционально) конкретный лист.
2. Система запускает preprocessing и формирует clean/raw таблицы.
3. Система обучает модель (если нужно) и выполняет анализ эпизодов.
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
