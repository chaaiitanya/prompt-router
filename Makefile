# Makefile — common development commands
# Run any target with: make <target>

.PHONY: run dev test lint install stack-up stack-down stack-logs

# ── Local development ─────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt ruff

run:
	uvicorn app.main:app --reload --port 8000

dev: run   # alias

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

lint:
	ruff check app/ tests/

# ── Docker stack ──────────────────────────────────────────────────────────────

stack-up:
	docker compose up --build -d

stack-down:
	docker compose down

stack-logs:
	docker compose logs -f gateway
