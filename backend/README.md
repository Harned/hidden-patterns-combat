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

## Известные ограничения (MVP)

- БД: SQLite через `create_all`. Если в локальном dev-режиме у вас уже
  была `storage/app.db` до TASK_SPEC_003, удалите её — схема расширилась
  полем `sources.mapping_config`. Alembic-миграция появится при переходе
  на Postgres.
- Анализ запускается **синхронно** в HTTP-запросе. Для больших файлов в будущем
  нужен фоновый воркер (Celery/Dramatiq/RQ).
- Пользователь получает только последний `AnalysisRun` — история не выдаётся.
- Не реализованы: смена пароля, подтверждение email, roles, rate-limit.
  Это намеренно отложено до post-MVP.
- `mapping_config` валидируется через `hpc_algo.ColumnMappingConfig` —
  единая точка доверия. Если pydantic отвергнет payload, клиент получит
  HTTP 422 с текстом ошибки.
