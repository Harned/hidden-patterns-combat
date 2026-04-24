# hpc-backend — FastAPI backend

Оркестрация auth, загрузок и вызовов независимого processing module
`hpc_algo`. **Никакой исследовательской логики** в backend нет.

## Области ответственности

- регистрация / вход / сессия (HttpOnly-cookie + JWT);
- загрузка Excel, проверка расширения и магических байт;
- хранение источников на локальной ФС по `storage/uploads/<user_id>/`;
- запуск `hpc_algo.analyze_source(...)` и сохранение `AnalysisResult` в SQLite;
- выдача последнего результата пользователю.

## Установка

Из корня репозитория:

```bash
source .venv/bin/activate
pip install -e "algo[dev]" -e "backend[dev]"
```

## Запуск dev-сервера

```bash
uvicorn app.main:app --reload --app-dir backend
```

OpenAPI: http://127.0.0.1:8000/docs

## Основные эндпоинты

| Метод | Путь                               | Описание                           |
|------:|------------------------------------|------------------------------------|
| POST  | `/api/auth/register`               | регистрация (email + password)     |
| POST  | `/api/auth/login`                  | вход                               |
| POST  | `/api/auth/logout`                 | выход                              |
| GET   | `/api/auth/me`                     | текущий пользователь               |
| GET   | `/api/sources`                     | список моих источников             |
| POST  | `/api/sources`                     | загрузить `.xlsx` / `.xls`         |
| GET   | `/api/sources/{id}`                | карточка источника                 |
| DELETE| `/api/sources/{id}`                | удалить                            |
| POST  | `/api/sources/{id}/preflight`      | предложить ColumnMappingConfig     |
| GET   | `/api/sources/{id}/mapping`        | получить сохранённый mapping       |
| PUT   | `/api/sources/{id}/mapping`        | сохранить mapping (валидация)      |
| DELETE| `/api/sources/{id}/mapping`        | сбросить mapping                   |
| GET   | `/api/sources/{id}/sheets/{sheet}/columns` | полный список flatten-колонок листа |
| POST  | `/api/sources/{id}/analyze?mode=auto` | запустить анализ (mapping если есть). `mode` ∈ {`auto`, `detailed`, `basic`, `off`} |
| GET   | `/api/sources/{id}/result`         | последний результат                |

## Тесты

```bash
cd backend
pytest
ruff check .
```

Тесты покрывают: регистрацию, login, ограничение просмотра чужих источников,
валидацию upload (расширение, макросы), запуск анализа и проверку
инвариантов `AnalysisResult` (observations = ЗАП, HMM-поля отсутствуют).

## Инфраструктура (TASK_SPEC_006)

- **Alembic** (`backend/alembic`) — первая миграция `0001_initial`
  описывает `users / sources / analysis_runs` и все текущие поля
  (`mapping_config`, `state`, `hmm_mode`, `started_at`, `finished_at`,
  `error`). Применить: `HPC_DATABASE_URL=... make db-upgrade`. В
  docker-compose миграции применяются автоматически при старте.
- **Postgres** — основная БД в проде (через psycopg3). SQLite остаётся
  для dev и тестов.
- **Фоновый analyze** — через FastAPI `BackgroundTasks`. По умолчанию
  `POST /analyze` → `state=pending`, результат доступен через
  `GET /runs/{run_id}`. `?wait=true` — синхронный режим для CLI/тестов.
- **CSRF** — double-submit cookie `hpc_csrf` + заголовок `X-CSRF-Token`.
  Включается `HPC_CSRF_REQUIRED=true` (dev/test — `false`).
- **Rate-limit** — простой in-memory token bucket для `/auth/login` и
  `/auth/register`; лимит задаётся `HPC_RATE_LIMIT_AUTH_PER_MINUTE`
  (по умолчанию 5). Включается `HPC_RATE_LIMIT_ENABLED=true`.

## Известные ограничения

- **Rate-limit in-memory** — не подходит для нескольких воркеров
  (нужен Redis-based).
- **Background через `BackgroundTasks`** выполняется в том же процессе.
  При высокой нагрузке — мигрировать на RQ/Dramatiq.
- **`mapping_config` валидируется через `hpc_algo.ColumnMappingConfig`** —
  единая точка доверия к формату. Pydantic отклоняет невалидный payload
  с 422.
- **Email без верификации**, нет 2FA и смены пароля.
- **Пользователь видит только последний `done`-run** (через
  `GET /result`). Timeline истории — UX-этап.
