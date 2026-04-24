# TASK_SPEC_006_INFRA

## Текущая задача

Подготовить прод-инфраструктуру для `hidden-patterns-combat`:

1. **PostgreSQL** как основная БД (с сохранением SQLite для dev-тестов).
2. **Alembic**-миграции (первая миграция описывает текущее состояние
   схемы, включая `sources.mapping_config` и таблицы
   `analysis_runs`/`users`).
3. **docker-compose** с сервисами `db`, `backend`, `frontend`,
   healthcheck'ами и dev-переменными окружения.
4. **Фоновая очередь** для `analyze` (чтобы большие Excel не блокировали
   HTTP-worker). Используем легковесный in-process worker через
   `fastapi.BackgroundTasks` + статус `AnalysisRun.state`
   (`pending` / `running` / `done` / `failed`). Полноценный RQ/Dramatiq —
   отдельный шаг, если потребуется горизонтальное масштабирование.
5. **CSRF** для мутирующих эндпоинтов (`POST` / `PUT` / `DELETE`).
   Double-submit cookie pattern: backend выдаёт `csrf_token` при
   login/register и проверяет его в заголовке `X-CSRF-Token` для
   мутирующих запросов. `GET` не проверяется.
6. **Rate-limit** на `/api/auth/register` и `/api/auth/login` (минимум:
   5 попыток в минуту на IP). Реализация — простая in-memory реализация
   token bucket, без внешних зависимостей; для прода позже можно
   подменить на Redis-based.

## Что НЕ делает этот шаг

* Не переводит на Redis / Celery.
* Не реализует полноценный audit-log.
* Не добавляет email-подтверждение / 2FA.
* Не меняет доменную логику алгоритма.

## Инварианты (неизменны)

* observations = ЗАП.
* скрытые состояния — доменные (TASK_SPEC_004 / TASK_SPEC_005).
* Алгоритм остаётся независимым модулем.
* Никакой исследовательской логики в backend/фронте.

## Формат миграции

* Alembic инициализируется с `script_location = backend/alembic` и
  `sqlalchemy.url` из env `HPC_DATABASE_URL`.
* Первая миграция `0001_initial` генерируется вручную (через
  `alembic revision` + скрипт `--autogenerate` на SQLite) и описывает
  таблицы `users`, `sources`, `analysis_runs` в текущем виде.
* Команда `alembic upgrade head` применяется при `docker-compose up`.
* На dev-SQLite миграция тоже работает; в `Settings` хранится флаг
  `use_alembic`; по умолчанию — `True` в проде, `False` в тестах
  (где используется `init_schema` с `create_all`).

## Формат docker-compose

Сервисы:

* `db` — `postgres:16-alpine`, volume `pgdata`, healthcheck
  `pg_isready`.
* `backend` — сборка из `backend/Dockerfile` (Python 3.11 + algo
  editable install + uvicorn), depends_on `db.healthy`.
* `frontend` — сборка из `frontend/Dockerfile` (node 20 build → nginx
  serve static), depends_on `backend`.

Env:

* `HPC_DATABASE_URL=postgresql+psycopg://hpc:hpc@db:5432/hpc`
* `HPC_SECRET_KEY` — обязательный (валидация в `Settings`).
* `HPC_ALLOWED_ORIGINS` — список доменов.

## Фоновая задача `analyze`

* `AnalysisRun.state: pending | running | done | failed`,
  `error: str | None`, `started_at`, `finished_at`.
* `POST /api/sources/{id}/analyze` создаёт `AnalysisRun(state=pending)`
  и ставит задачу через `BackgroundTasks`.
* `GET /api/sources/{id}/runs/{run_id}` возвращает текущее состояние
  run'а.
* `GET /api/sources/{id}/result` по-прежнему возвращает последний
  завершённый run (с `state=done`).

## CSRF

* После `/api/auth/register` и `/api/auth/login` backend выдаёт cookie
  `csrf_token` (без HttpOnly — фронту нужен JS-доступ) и возвращает
  значение в теле ответа (`UserPublic.csrf_token`).
* Мутирующие запросы должны иметь заголовок `X-CSRF-Token` равный
  значению cookie.
* В тестах проверка CSRF отключена флагом `settings.csrf_required` (в
  dev/test — `False`), в production — `True`.

## Rate-limit

* Простая in-memory реализация (token bucket на IP).
* Применяется только к `/api/auth/register` и `/api/auth/login`.
* Превышение → 429.

## Definition of Done

1. Alembic: `make db-upgrade` обновляет схему. Первая миграция
   воспроизводит текущие таблицы.
2. `docker-compose up` поднимает всё локально; e2e сценарий
   register → upload → analyze → result проходит.
3. Фоновая задача: `analyze` возвращает `202 Accepted` с `run_id` и
   `state=pending`; через несколько секунд статус = `done`.
4. CSRF: мутирующий запрос без заголовка `X-CSRF-Token` получает
   `403` в production-режиме.
5. Rate-limit: 6-й подряд запрос `login` получает `429`.
6. Все текущие тесты продолжают проходить, `ruff`/`tsc` clean.
7. README обновлён; приватные данные по-прежнему НЕ коммитятся.
