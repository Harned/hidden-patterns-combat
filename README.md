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

## Точка входа (CLI)
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

## Выходные артефакты анализа
- `artifacts/analysis/episode_analysis.csv`
- `artifacts/analysis/state_profile.csv`
- `artifacts/analysis/interpretation.txt`
- `artifacts/analysis/report.md`
- `artifacts/analysis/hidden_states.png` (если доступна визуализация)
- `artifacts/analysis/state_probabilities.png` (если доступна визуализация)
