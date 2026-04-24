# hidden-patterns-combat

Анализ соревновательной деятельности спортсменов-единоборцев по
Excel-источникам. На первом этапе — независимый исследовательский алгоритм
с honest-baseline; на втором — веб-приложение с регистрацией, загрузкой
Excel, списком источников и русскоязычным интерфейсом анализа.

## Предметные инварианты (неперего­вариваемые)

- `observations = ЗАП`;
- скрытые состояния следуют цепочке
  `маневрирование -> КФВ -> ВУП -> ЗАП`;
- алгоритм — независимый модуль обработки;
- backend и frontend не содержат исследовательской логики;
- при недостатке данных возвращается честный `baseline / audit / warnings`,
  а не фейковая HMM.

Контекст: см. `docs/agent_context/` (`DOMAIN_SPEC.md`, `AGENT_RULES.md`,
`ALGORITHM_SPEC.md`, `DEFINITION_OF_DONE.md`).

## Статус проекта

- [x] **TASK_SPEC_001** — архитектура согласована.
- [x] **TASK_SPEC_002 / Этап 1** — independent `algo/`: honest audit,
  детекция колонок, baseline-распределения, chart-JSON, CLI, тесты.
- [x] **TASK_SPEC_002 / Этап 2** — backend (FastAPI + SQLite + JWT-cookie):
  auth, upload, sources, analyze, persistence; e2e-тесты.
- [x] **TASK_SPEC_002 / Этап 3** — frontend (React + Vite + TS + Tailwind +
  TanStack Query + Recharts): русскоязычный UI, левая панель источников,
  центральная область с AnalysisResult.
- [x] **TASK_SPEC_003** — column mapping: multi-row header, preflight с
  эвристическим распределением ролей, сохранение config в БД, UI-редактор
  «колонка → роль», перезапуск анализа с сохранённым mapping. На реальном
  `docs/Оценка СД содержание.xlsx` статус переходит из
  `needs_column_mapping` в `baseline_only` с непустыми распределениями по
  всем четырём предметным группам (маневрирование / КФВ / ВУП / ЗАП) и
  time-статистикам.
- [x] **TASK_SPEC_003_1** — корректная кодировка ЗАП. Алгоритм
  классифицирует ЗАП-колонки как `categorical` / `binary` / `count` /
  `empty` и считает события по каналам (`zap_events_by_channel`,
  `zap_total_triggers`). На реальных данных ЗАП теперь выражается в
  осмысленных каналах (Удержание=44, На ногу=32, На руку=14), а не в
  гистограммах 0/1.
- [x] **TASK_SPEC_003_2** — в `MappingEditor` доступен полный список
  flatten-колонок листа с sample values и `role_hint`. Новый эндпоинт
  `GET /api/sources/{id}/sheets/{sheet}/columns?header_rows=…`
  возвращает все 70 колонок реального Excel; пользователь может
  переназначить роль любой колонке. Также исправлена коллизия TIME vs
  EPISODE (колонка «Время эпизода, с.» теперь корректно идёт в `time`).
- [x] **TASK_SPEC_004** — HMM-ветка. 3-state модель с предметными именами
  `маневрирование / КФВ / ВУП`, observations строятся из ЗАП-каналов,
  доменно информированная инициализация `π` и `A`, воспроизводимый seed.
  Запуск разрешён только после прохождения guard'ов (min эпизодов,
  min ЗАП-событий, min алфавит, mapping) и sanity-check матрицы
  переходов; иначе возвращается честный `baseline_only` + warning
  `hmm.guards_failed`. На реальном `docs/Оценка СД содержание.xlsx`
  получили `status=hmm_ready` (204 эпизода, state_distribution =
  `маневрирование=0.80, КФВ=0.14, ВУП=0.06` — совпадает с доменной
  цепочкой).
- [x] **TASK_SPEC_005** — детализированная 7-state HMM с доменными
  именами `маневры / захваты / хваты / обхваты / прихваты / упоры /
  ВУП`. Режимы запуска: `auto` / `detailed` / `basic` / `off`;
  `auto` выбирает 7-state только при лучшем BIC, иначе возвращает
  3-state. Дополнительные guard'ы для detailed (120 эпизодов, 60
  ЗАП-событий, 3+ токена в алфавите, 5+ состояний использованы).
  На реальных данных `auto` корректно фоллбэчит в `basic_3state`
  (alphabet на сегодняшних файлах слишком мал для 7 состояний), что
  методологически правильно.

## Структура репозитория

```
hidden-patterns-combat/
├── algo/                # independent processing module (Python, hpc_algo)
│   ├── hpc_algo/        # api.py, schema.py, loading, audit, detection, baseline, cli
│   ├── tests/           # unit-тесты на synthetic Excel
│   └── pyproject.toml
├── backend/             # FastAPI + SQLAlchemy + SQLite
│   ├── app/             # auth, sources, analysis, db, config, main
│   ├── tests/           # httpx TestClient: auth/sources/analysis
│   └── pyproject.toml
├── frontend/            # React + Vite + TypeScript (русскоязычный UI)
│   ├── src/             # pages, features/{sources,analysis}, auth, api, components
│   └── package.json
├── docs/
│   ├── agent_context/   # обязательный контекст для любого агента
│   └── Оценка СД содержание.xlsx  # реальный Excel-источник
├── storage/             # .gitignore: uploaded files + sqlite app.db (dev)
├── Makefile             # единые команды
└── README.md
```

## Быстрый старт

Требования: Python 3.11+, Node.js 20+, npm 10+.

```bash
# 1. Python-пакеты (algo + backend)
make install

# 2. Зависимости фронта
make install-frontend

# 3. Запустить backend (http://127.0.0.1:8000, OpenAPI на /docs)
make dev-backend

# 4. В другом терминале — фронтенд (http://127.0.0.1:5173)
make dev-frontend
```

## Проверка

```bash
make test    # 16 algo + 13 backend тестов
make lint    # ruff (algo + backend) + tsc (frontend)
```

CLI напрямую по реальному Excel:

```bash
make report    # короткий человекочитаемый отчёт
make summary   # компактная сводка (JSON)
make analyze   # полный AnalysisResult -> .local/result.json
```

## End-to-end smoke (проверено)

Полный цикл на реальном `docs/Оценка СД содержание.xlsx`:

**Без column mapping** (архитектурно безопасный fallback):

1. register → upload → analyze → `status = needs_column_mapping`;
2. результат содержит `audit.possible_multirow_header` и указание, что
   нужен TASK_SPEC_003.

**С column mapping (TASK_SPEC_003, через API)**:

1. `POST /api/auth/register` + upload Excel (709 KB) → 201.
2. `POST /api/sources/{id}/preflight` → авто-сопоставление для всех 13
   листов: маневрирование=24, КФВ=29, ВУП=5, ЗАП=3 (совпадает с
   `DATA_SPEC.md`), `header_rows = [1, 2, 3]`.
3. `PUT /api/sources/{id}/mapping` (с payload preflight) → 200,
   `has_mapping=true`.
4. `POST /api/sources/{id}/analyze` → `status = baseline_only`.
5. `GET /api/sources/{id}/result`:
   - `applied_mapping` присутствует;
   - `hidden_group_totals`: `маневрирование=69720, КФВ=84257,
     ВУП=14525, episode=5785, ЗАП=90` (ЗАП = число реальных событий
     после TASK_SPEC_003_1, а не сумма 0/1);
   - `zap_events_by_channel`: `{Удержание: 44, На ногу: 32, На руку: 14}` —
     это именно observations, перечисленные в `DOMAIN_SPEC.md`;
   - `zap_column_kinds`: 27 binary + 12 count-колонок;
   - `episodes_per_sheet`: 10–21 эпизод на каждую весовую категорию;
   - `time_statistics` для 13 колонок;
   - HMM-поля отсутствуют (инвариант сохранён).

## Известные проблемы и ограничения

Перечислены явно, чтобы решать их после MVP, а не прятать.

1. **Схема ЗАП-кодировки.** Закрыто в `TASK_SPEC_003_1` +
   `TASK_SPEC_004`: алгоритм классифицирует ЗАП-колонки на
   `categorical / binary / count / empty`, а HMM строит
   последовательности по каналам (`Удержание`, `На руку`, `На ногу`,
   `ЗАП-Р` и т.п.) c токеном `_noop_` для «пустых» эпизодов.
   Открытые: более богатые эмиссии (multivariate вместо Categorical),
   детализированная 7-state модель (маневры / захваты / хваты /
   обхваты / прихваты / упоры / ВУП).
2. **Миграция схемы БД.** `TASK_SPEC_003` добавил `sources.mapping_config`.
   В dev достаточно удалить `storage/app.db` — `create_all` создаст схему
   заново. Для прода при переходе на Postgres нужен Alembic.
3. **Синхронный анализ в HTTP-запросе.** Большие Excel могут блокировать
   uvicorn-worker. Для прод — фоновый воркер (RQ/Dramatiq/Celery).
3. **Alembic не настроен.** Для MVP используется `Base.metadata.create_all`.
   При переходе на Postgres — добавить Alembic и первую миграцию.
4. **История запусков скрыта.** API и UI отдают только последний
   `AnalysisRun`. История нужна отдельным шагом.
5. **Нет rate-limit и CSRF защит.** Для dev/MVP это приемлемо, для прода —
   добавить `slowapi`/reverse-proxy + CSRF-токен для non-idempotent операций.
6. **Email без верификации.** Пароль ≥ 8 символов, bcrypt, JWT в HttpOnly,
   но подтверждение email и восстановление пароля отложены.
7. **`passlib[bcrypt]` предупреждает о `crypt` в Python 3.13.** Это
   зависимость `passlib`; при апгрейде Python >=3.13 заменить на
   `bcrypt`-only адаптер.
8. **Frontend bundle 599 KB** (Recharts + React + TanStack). Для MVP ok;
   code-split по маршрутам — задача следующего этапа.
9. **Тестов фронтенда нет.** Только TypeScript + Vite build. Добавить
   Vitest + Testing Library после стабилизации UI.
10. **Нет docker-compose.** Пока всё поднимается через `make dev-backend` /
    `make dev-frontend`. Контейнеризацию соберу, когда будем переходить на
    Postgres.
11. **UI группирует `audit.suspicious`, но на реальном Excel warnings всё
    равно многословны.** Это следствие реальной структуры данных; после
    column mapping шум уйдёт сам.
12. **CORS настроен только для `http://localhost:5173`.** Для прода
    перечень Origin'ов нужно вынести в `HPC_ALLOWED_ORIGINS` env.

## Архитектурные границы (напоминание)

- `algo/` не импортирует ничего из `backend`/`frontend`.
- `backend/` импортирует только `hpc_algo.analyze_source`,
  не работает с `pandas`/`numpy` напрямую.
- `frontend/` рендерит `AnalysisResult` как есть, интерпретация — на
  стороне алгоритма.
- Любой кандидат колонки — **подсказка**. Подтверждение column mapping —
  за пользователем.

## Что делать дальше

- **TASK_SPEC_005 (детализированная HMM)**: 7-state модель с
  вложенной структурой (маневры / захваты / хваты / обхваты / прихваты
  / упоры / ВУП) и более богатой эмиссией.
- **TASK_SPEC_006 (продовая инфраструктура)**: Postgres + Alembic
  (миграция для `sources.mapping_config`, `sources.*`), docker-compose,
  фоновые задачи (запуск `analyze` в воркере), rate-limit, CSRF.
- **UX-улучшения**: превью данных на листе рядом с редактором mapping;
  fuzzy-search по колонкам; подсветка повторного использования роли;
  визуализация Viterbi-траекторий как timeline на лист/эпизод.
- **Методологический пункт**: проверить HMM-результат на
  cross-validation по весовым категориям (обучение на одних, оценка на
  других) — когда появится достаточно реальных источников.
