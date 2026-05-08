PYTHON ?= python3.11
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
REAL_EXCEL ?= docs/Оценка СД содержание.xlsx

.PHONY: help venv install install-backend install-frontend \
        test test-algo test-backend \
        lint lint-algo lint-backend typecheck-frontend \
        analyze report summary individual-models \
        dev-backend dev-frontend build-frontend \
        db-upgrade db-revision \
        docker-up docker-down docker-logs \
        clean

help:
	@echo "Setup:"
	@echo "  make venv                 — создать Python $(PYTHON) venv"
	@echo "  make install              — установить algo + backend (editable, dev-extras)"
	@echo "  make install-frontend     — npm install в frontend/"
	@echo ""
	@echo "Dev:"
	@echo "  make dev-backend          — uvicorn на http://127.0.0.1:8000"
	@echo "  make dev-frontend         — Vite на http://127.0.0.1:5173 (proxy /api)"
	@echo "  make build-frontend       — prod-сборка SPA в frontend/dist"
	@echo ""
	@echo "Database (Alembic):"
	@echo "  HPC_DATABASE_URL=... make db-upgrade   — применить миграции"
	@echo "  MSG=... make db-revision               — создать новую миграцию"
	@echo ""
	@echo "Docker (prod-like local stack):"
	@echo "  make docker-up            — build + up (db + backend + frontend)"
	@echo "  make docker-down          — stop + remove"
	@echo "  make docker-logs          — follow logs"
	@echo ""
	@echo "Quality:"
	@echo "  make test                 — pytest algo/ + backend/"
	@echo "  make lint                 — ruff algo/ + backend/ + tsc frontend/"
	@echo ""
	@echo "Algo CLI on real Excel:"
	@echo "  make analyze / report / summary"
	@echo "  make individual-models    — TASK_SPEC_011: ~30 HTML-моделей в reports/individual/"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip

install: venv
	$(VENV_PIP) install -e "algo[dev]" -e "backend[dev]"

install-backend: venv
	$(VENV_PIP) install -e "algo[dev]" -e "backend[dev]"

install-frontend:
	cd frontend && npm install

# --- Quality ---

test-algo:
	cd algo && ../$(VENV_PY) -m pytest

test-backend:
	cd backend && ../$(VENV_PY) -m pytest

test: test-algo test-backend

lint-algo:
	cd algo && ../$(VENV_PY) -m ruff check .

lint-backend:
	cd backend && ../$(VENV_PY) -m ruff check .

typecheck-frontend:
	cd frontend && npm run typecheck

lint: lint-algo lint-backend typecheck-frontend

# --- Dev servers ---

dev-backend:
	$(VENV)/bin/uvicorn app.main:app --reload --app-dir backend

dev-frontend:
	cd frontend && npm run dev

build-frontend:
	cd frontend && npm run build

# --- Alembic (requires HPC_DATABASE_URL env) ---

db-upgrade:
	cd backend && ../$(VENV_PY) -m alembic -c alembic.ini upgrade head

db-revision:
	cd backend && ../$(VENV_PY) -m alembic -c alembic.ini revision --autogenerate -m "$(MSG)"

# --- Docker compose (prod-like local stack) ---

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=100

# --- Algo CLI on real Excel ---

analyze:
	@mkdir -p .local
	$(VENV)/bin/hpc-algo analyze "$(REAL_EXCEL)" --output .local/result.json

report:
	$(VENV)/bin/hpc-algo report "$(REAL_EXCEL)"

summary:
	$(VENV)/bin/hpc-algo summary "$(REAL_EXCEL)"

# TASK_SPEC_011 — индивидуальные 5-state Marков-модели.
# Источник: $(REAL_EXCEL); конфиг: $(STATE_GROUPS); выход: $(INDIVIDUAL_OUT_DIR).
STATE_GROUPS ?= config/state_groups.yaml
INDIVIDUAL_OUT_DIR ?= reports/individual

individual-models:
	@mkdir -p "$(INDIVIDUAL_OUT_DIR)"
	$(VENV)/bin/hpc-algo individual-markov "$(REAL_EXCEL)" \
		--state-groups "$(STATE_GROUPS)" \
		--output-dir "$(INDIVIDUAL_OUT_DIR)"

clean:
	rm -rf $(VENV) .pytest_cache algo/.pytest_cache algo/**/__pycache__ \
	       backend/.pytest_cache backend/**/__pycache__ \
	       .ruff_cache algo/.ruff_cache backend/.ruff_cache \
	       frontend/node_modules frontend/dist \
	       .local storage
