.PHONY: help install phoenix phoenix-down prepare check smoke run-main report sync audit test coverage lint typecheck validate

SYSTEMS ?= jev baseline baseline-conf baseline-2
# One run of every message. Repeats of the same messages would make the report's margins of error too narrow.
REPEATS ?= 1
# Optional: RUN_ID=... on the command line. Recipes default to a new UTC timestamp (runs) or the latest run (report/sync).
export RUN_ID

help: ## Show targets
	@grep -E '^[a-z-]+:.*## ' Makefile | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dependencies (uses the OS trust store for corporate TLS)
	uv sync --system-certs

phoenix: ## Start Phoenix (UI + collector) and register Jev's token price for cost dashboards
	docker compose up -d phoenix
	scripts/phoenix_up.sh

phoenix-down: ## Stop Phoenix (data is kept in the phoenix_data volume)
	docker compose down

prepare: ## Build data/splits/*.jsonl from the raw CSV
	uv run jevbench prepare

check: ## One live call per system to validate credentials
	uv run jevbench check --systems $(SYSTEMS)

smoke: ## 5 examples x every system on the practice split, then report
	scripts/run_matrix.sh "smoke-$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}" "practice" "$(SYSTEMS)" "1" --limit 5 --warmup 2

run-main: ## Full experiment: every system x main + profanity splits (needs committed code)
	scripts/run_matrix.sh "$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}" "main profanity" "$(SYSTEMS)" "$(REPEATS)"

report: ## Report the latest run (or RUN_ID=...)
	uv run jevbench report $${RUN_ID:+--run-id "$$RUN_ID"}

sync: ## Mirror the latest run (or RUN_ID=...) into Phoenix
	uv run jevbench phoenix-sync $${RUN_ID:+--run-id "$$RUN_ID"}

audit: ## Before publishing: scan every file git would commit for the gateway host, API keys, and .audit-denylist
	uv run jevbench audit

test: ## Offline unit tests
	uv run pytest -q

coverage: ## Offline unit tests with a coverage report (fails under 85%)
	uv run pytest -q --cov --cov-report=term-missing

lint: ## Ruff lint + format check, YAML lint
	uv run ruff check src tests && uv run ruff format --check src tests
	uv run yamllint -c .yamllint.yaml data/intents.yaml docker-compose.yml .yamllint.yaml .github/workflows

typecheck: ## Pyright
	uv run pyright

validate: lint typecheck coverage ## Everything CI runs
