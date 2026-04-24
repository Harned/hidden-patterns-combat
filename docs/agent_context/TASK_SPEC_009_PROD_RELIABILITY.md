# TASK_SPEC_009_PROD_RELIABILITY

## Текущая задача

Укрепить прод-инфраструктуру до уровня, приемлемого для пилотного
запуска: горизонтальное масштабирование бэкенда перестаёт ронять
rate-limit; учётная запись пользователя подтверждается email'ом; время
жизни JWT короткое, с refresh-токеном.

## Что должно появиться

1. **Rate-limit с pluggable backend**:
   * фабрика выбирает `RedisRateLimiter`, если задан
     `HPC_RATE_LIMIT_BACKEND=redis` и `HPC_REDIS_URL`,
     иначе — in-memory как раньше;
   * Redis-адаптер использует атомарный `INCR` + `EXPIRE`, ключ
     `rl:{bucket}:{ip}:{minute}`;
   * если `redis` пакет не установлен или соединение не отвечает —
     backend логирует ошибку и падает к in-memory (в продакшене логи
     сигнализируют, что нужен Redis).
2. **Email verification (заглушка отправки)**:
   * `User.email_verified_at: datetime | None`;
   * `GET /api/auth/verify-email?token=...` — подтверждает адрес;
   * `POST /api/auth/request-verification` — выдаёт токен-ссылку
     (логируем в stdout; реальный SMTP — отдельная задача);
   * поле `email_verified_at` в ответе `/auth/me`;
   * Alembic `0002_email_verified_at` добавляет колонку.
3. **Refresh-токены**:
   * access-token: 15 мин; refresh-token: 14 дней;
   * в JWT `typ: access | refresh`;
   * `POST /api/auth/refresh` обновляет cookie + CSRF.
4. **Конфиг**:
   * `HPC_ACCESS_TOKEN_EXPIRES_MINUTES=15`;
   * `HPC_REFRESH_TOKEN_EXPIRES_MINUTES=60*24*14`;
   * `HPC_RATE_LIMIT_BACKEND` ∈ {`memory`, `redis`}.

## Что НЕ делает этот шаг

* Не реализует реальную SMTP-отправку.
* Не вводит ролевой доступ.
* Не переезжает фоновую очередь на Redis/RQ — это отдельный шаг
  при росте нагрузки.

## Definition of Done

1. `HPC_RATE_LIMIT_BACKEND=redis` с доступным Redis поднимает
   Redis-лимитер; без Redis backend корректно возвращает 429 на
   in-memory уровне (логируется fallback).
2. Новый пользователь попадает в систему с `email_verified_at = null`;
   после `GET /auth/verify-email?token=...` поле заполнено.
3. По истечении access-токена клиент может вызвать `POST /auth/refresh`
   с refresh-cookie и получить новые access + csrf.
4. Тесты `algo + backend` зелёные; новая миграция применяется на
   SQLite и Postgres.
5. README обновлён, все новые env-переменные задокументированы.
