FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Сначала ставим зависимости algo + backend, чтобы сборка кэшировалась.
COPY algo /app/algo
COPY backend /app/backend

RUN pip install --upgrade pip \
 && pip install -e /app/algo \
 && pip install -e /app/backend

ENV PYTHONPATH=/app/backend
WORKDIR /app/backend

EXPOSE 8000

# Миграции + uvicorn. HPC_DATABASE_URL / HPC_SECRET_KEY — обязательны из env.
CMD ["sh", "-c", "alembic -c alembic.ini upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
