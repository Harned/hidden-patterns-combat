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
- [x] **TASK_SPEC_006** — продовая инфраструктура: Alembic-миграция
  `0001_initial` (users/sources/analysis_runs + `mapping_config`
  + расширенный `AnalysisRun` state/hmm_mode/started_at/finished_at/error);
  поддержка Postgres через `HPC_DATABASE_URL` + psycopg;
  `docker-compose.yml` (db + backend + frontend + healthchecks);
  фоновый analyze через FastAPI `BackgroundTasks` со статусами
  `pending → running → done | failed`; GET `/sources/{id}/runs/{run_id}`
  для опроса; double-submit CSRF (токен в non-HttpOnly cookie
  `hpc_csrf` + заголовок `X-CSRF-Token`, включается
  `HPC_CSRF_REQUIRED=true` в проде); простой in-memory rate-limit
  на `/auth/login` и `/auth/register` (5 попыток/минуту).
- [x] **TASK_SPEC_007** — UX: вкладка «История запусков» c опросом
  состояний и открытием прошлых `done`-run'ов
  (`GET /sources/{id}/runs` + `/runs/{id}/result`); preview первых
  строк листа в редакторе mapping
  (`GET /sources/{id}/sheets/{name}/preview`); визуализация
  Viterbi-траектории по эпизодам с цветовой легендой; code-split
  `MappingEditor` / `HMMView` через `React.lazy`; Vitest + Testing
  Library scaffolding (`npm run test:unit`); русская локализация
  warning-кодов (`translateWarning`).
- [x] **TASK_SPEC_008** — multivariate Bernoulli эмиссии и k-fold CV:
  `AnalyzeConfig.observation_emission ∈ {categorical, bernoulli}`;
  `hpc_algo/hmm_bernoulli.py` с собственной log-space реализацией
  forward-backward/Viterbi/EM (без SciPy); `hpc_algo/cv.py` с
  `cross_validate(frames, config, run_config, k)` поверх листов —
  честное сравнение вариантов модели.

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

### Dev (SQLite, без Docker)

```bash
make install
make install-frontend
make dev-backend      # uvicorn на http://127.0.0.1:8000
make dev-frontend     # Vite на http://127.0.0.1:5173 (proxy /api)
```

### Prod-like (Docker Compose)

```bash
# HPC_SECRET_KEY — обязательный секрет (≥ 32 байт), остальное по умолчанию
HPC_SECRET_KEY="$(openssl rand -hex 32)" make docker-up
# → frontend на http://127.0.0.1:8080, backend на 8000, Postgres на 5432
```

Миграции применяются автоматически при старте backend-контейнера. Для
ручного прогона против внешней БД:

```bash
HPC_DATABASE_URL="postgresql+psycopg://user:pass@host/db" make db-upgrade
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
2. **Миграция схемы БД.** Закрыто в `TASK_SPEC_006`: есть
   Alembic + Postgres; dev-SQLite всё ещё работает через `create_all`
   (если `HPC_USE_ALEMBIC=false`).
3. **Фоновый анализ.** Закрыто в `TASK_SPEC_006`: `POST
   /sources/{id}/analyze` асинхронный (возвращает `state=pending`),
   `GET /sources/{id}/runs/{run_id}` отдаёт прогресс. Для CLI и тестов
   остаётся синхронный режим `?wait=true`. Распределённая очередь
   (RQ/Dramatiq) — только если потребуется горизонтальное
   масштабирование.
4. **CSRF + rate-limit.** Закрыто в `TASK_SPEC_006`. Включаются env-
   флагами (`HPC_CSRF_REQUIRED`, `HPC_RATE_LIMIT_ENABLED`) — в
   docker-compose они `true` по умолчанию.
5. **docker-compose.** Закрыто в `TASK_SPEC_006`: `make docker-up`
   поднимает db + backend + frontend с healthcheck'ами.
6. **История запусков пока не выводится в UI.** API `/runs/{run_id}`
   есть, но интерфейс показывает только последний `done`-run. Timeline
   истории — задача UX-этапа.
7. **Email без верификации.** Пароль ≥ 8 символов, bcrypt, JWT в
   HttpOnly, но подтверждение email и восстановление пароля отложены.
8. **`passlib[bcrypt]` предупреждает о `crypt` в Python 3.13.** Это
   зависимость `passlib`; при апгрейде Python >=3.13 заменить на
   `bcrypt`-only адаптер.
9. **Frontend bundle ≈615 KB** (Recharts + React + TanStack). Code-split
   по маршрутам — оптимизация следующего этапа.
10. **Тестов фронтенда нет.** Только TypeScript + Vite build. Vitest +
    Testing Library — отдельный UX-этап.
11. **Rate-limit in-memory.** Не подходит для multi-worker. Для прода
    на нескольких воркерах заменить на Redis-based реализацию.
12. **CORS**. Настраивается через `HPC_ALLOWED_ORIGINS` env (список
    origin'ов через запятую).

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
