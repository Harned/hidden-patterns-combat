# hpc-algo — independent processing module

Независимый модуль обработки Excel-источников для проекта
`hidden-patterns-combat`.

## Область ответственности

Модуль делает и ничего больше:

- читает Excel (`.xlsx` / `.xls` / `.xlsm`), включая **многострочные
  заголовки** (`header_rows=[0,1,2]` и т.п.);
- собирает **честный аудит**: листы, размеры, колонки, dtypes, пропуски,
  подозрительные значения, preview;
- эвристически определяет кандидатов на предметные группы
  (`ЗАП`, `маневрирование`, `КФВ`, `ВУП`, `time`, `athlete`, ...) —
  как на обычных заголовках, так и на flatten-именах multi-row header;
- выполняет **preflight** `ColumnMappingConfig` на основе flatten-заголовков
  (TASK_SPEC_003);
- при подтверждённом column mapping считает baseline-распределения **по
  всем четырём предметным группам** и time-статистики;
- классифицирует ЗАП-колонки как `categorical / binary / count / empty`
  и отдаёт `zap_events_by_channel` и `zap_total_triggers`
  (TASK_SPEC_003_1), чтобы пользователь видел события по каналам
  `Удержание / На руку / На ногу / ЗАП-Р / …`, а не гистограмму 0/1;
- при выполнении всех guard'ов качества (min эпизодов, min ЗАП-событий,
  min алфавит, sanity-check матрицы переходов) **обучает 3-state HMM**
  с предметными состояниями `маневрирование / КФВ / ВУП` и возвращает
  `HMMResult` (A, B, π, per-episode Viterbi path, state distribution,
  interpretation). При любом нарушении guard'а — `status=baseline_only`
  + честный warning `hmm.guards_failed`;
- готовит chart-ready JSON для фронтенда;
- возвращает понятные `warnings` и честный `status`
  (`audit_only`, `baseline_only`, `needs_column_mapping`, `failed`).

Модуль **не** делает:

- не строит HMM без подтверждённого column mapping и без достаточного
  объёма данных (guard'ы обязательны);
- не возвращает HMM-поля в `AnalysisResult`, если `status != hmm_ready`;
- не знает про HTTP, БД, аутентификацию, UI;
- не подменяет `ЗАП` действиями спортсмена;
- не выдаёт диагностический вывод без оснований.

См. `docs/agent_context/DOMAIN_SPEC.md` и `docs/agent_context/ALGORITHM_SPEC.md`
для предметных инвариантов.

## Установка (локально)

Требуется Python 3.11+. Из корня репозитория:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e "algo[dev]"
```

## Запуск из CLI

Полный анализ в JSON (без column mapping — fallback):

```bash
hpc-algo analyze "docs/Оценка СД содержание.xlsx" --output result.json
```

Предложить ColumnMappingConfig (preflight):

```bash
hpc-algo preflight "docs/Оценка СД содержание.xlsx" --output mapping.json
```

Анализ с заранее подготовленным mapping:

```bash
hpc-algo analyze-with-mapping "docs/Оценка СД содержание.xlsx" mapping.json \
    --output mapped_result.json
```

Компактная сводка / человекочитаемый отчёт:

```bash
hpc-algo summary "docs/Оценка СД содержание.xlsx"
hpc-algo report  "docs/Оценка СД содержание.xlsx"
```

## Запуск из Python / notebook

```python
from hpc_algo import AnalyzeConfig, analyze_source, preflight_mapping

# 1. Честный audit-only / baseline
result = analyze_source("docs/Оценка СД содержание.xlsx")
print(result.status)

# 2. Preflight mapping (эвристика)
mapping = preflight_mapping("docs/Оценка СД содержание.xlsx")

# 3. Анализ с mapping
result2 = analyze_source(
    "docs/Оценка СД содержание.xlsx",
    AnalyzeConfig(column_mapping=mapping),
)
print(result2.status)
print(result2.basic_statistics.hidden_group_totals)
```

## Тесты и линт

```bash
cd algo
pytest
ruff check .
```

В unit-тестах используются *только* synthetic Excel-фикстуры. Реальный
`docs/Оценка СД содержание.xlsx` проверяется вручную через CLI — по правилам
`AGENT_RULES.md` синтетика не служит основным сценарием проверки модели.

## Структура пакета

```
algo/
├── pyproject.toml
├── hpc_algo/
│   ├── __init__.py      # публичный API
│   ├── api.py           # analyze_source(), preflight_mapping(), analysis_summary()
│   ├── schema.py        # AnalysisResult, AuditReport, BaselineReport,
│   │                    # ColumnMappingConfig, SheetMapping, TimeStats, ...
│   ├── loading.py       # чтение Excel, sha256, проверка расширения
│   ├── audit.py         # честный аудит: листы, колонки, типы, пропуски, preview
│   ├── detection.py     # эвристика кандидатов групп (ЗАП, КФВ, ВУП, ...)
│   ├── mapping.py       # flatten multi-row header + preflight-эвристика
│   ├── baseline.py      # baseline-распределения + chart-ready JSON
│   ├── hmm.py           # HMM-ветка (TASK_SPEC_004): guards, fit, Viterbi
│   ├── text_utils.py    # normalization кириллических заголовков
│   └── cli.py           # Typer CLI: analyze / preflight /
│                        # analyze-with-mapping / summary / report
└── tests/               # pytest: loading / audit / detection /
                         # mapping / analyze_source
```

## Контракт `AnalysisResult`

Единственный «мост» между модулем, backend и frontend. Поля верхнего уровня:

- `status` — честный статус (`audit_only` / `baseline_only` /
  `needs_column_mapping` / `hmm_ready` / `failed`). `hmm_ready` в MVP
  не выставляется никогда.
- `source_metadata` — имя файла, размер, sha256, список листов.
- `data_audit` — структурная статистика по листам.
- `detected_columns` — кандидаты, распознанные группы, недостающие группы,
  список явных предположений.
- `basic_statistics` — распределения ЗАП и пропусков.
- `charts` — chart-ready JSON без привязки к конкретной библиотеке.
- `warnings` / `errors` — коды, сообщения, severity, контекст.
- `report` — человекочитаемая сводка (ru).

## Известные ограничения

1. Детекция колонок — эвристика по заголовкам и небольшому семплу значений.
   Любой кандидат требует ручного подтверждения.
2. Многострочные заголовки (как в реальном `docs/Оценка СД содержание.xlsx`)
   сейчас распознаются как подозрительное наблюдение, но не разбираются
   автоматически. Это работа следующего этапа — `TASK_SPEC_003` (column mapping).
3. HMM-ветка в MVP отключена. Её включение требует отдельного
   подтверждения column mapping и валидации качества — см. `ALGORITHM_SPEC.md`.
