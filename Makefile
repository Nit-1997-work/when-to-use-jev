.PHONY: help install phoenix phoenix-down audit test coverage lint typecheck validate \
	intent-prepare intent-check intent-smoke intent-run intent-report intent-sync \
	rerank-grocery-candidates rerank-prepare rerank-check rerank-smoke rerank-practice rerank-tune rerank-run rerank-report

# Optional: RUN_ID=... on the command line. Recipes default to a new UTC timestamp (runs) or the latest run (report/sync).
export RUN_ID

# ---------- Experiment 1: intent classification ----------
INTENT_SYSTEMS ?= jev baseline baseline-conf baseline-2
# One run of every message. Repeats of the same messages would make the report's margins of error too narrow.
INTENT_REPEATS ?= 1

# ---------- Experiment 2: product search re-ranking ----------
RERANK_SYSTEMS ?= gemini-listwise gemini-flash-listwise gemini-label gemini-pointwise \
	jev-score jev-noul-batch jev-noul-pair jev-choice lexical random

help: ## Show targets
	@grep -E '^[a-z-]+:.*## ' Makefile | awk 'BEGIN {FS = ":.*## "}; {printf "  %-26s %s\n", $$1, $$2}'

install: ## Install dependencies (uses the OS trust store for corporate TLS)
	uv sync --system-certs

phoenix: ## Start Phoenix (UI + collector) and register Jev's token price for cost dashboards
	docker compose up -d phoenix
	scripts/phoenix_up.sh

phoenix-down: ## Stop Phoenix (data is kept in the phoenix_data volume)
	docker compose down

intent-prepare: ## Experiment 1: build data/intent/splits/*.jsonl from the raw CSV
	uv run jevbench intent prepare

intent-check: ## Experiment 1: one live call per system to validate credentials
	uv run jevbench intent check --systems $(INTENT_SYSTEMS)

intent-smoke: ## Experiment 1: 5 examples x every system on the practice split, then report
	scripts/run_matrix.sh intent "smoke-$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}" "practice" "$(INTENT_SYSTEMS)" "1" --limit 5 --warmup 2

intent-run: ## Experiment 1: every system x main + profanity splits (needs committed code)
	scripts/run_matrix.sh intent "$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}" "main profanity" "$(INTENT_SYSTEMS)" "$(INTENT_REPEATS)"

intent-report: ## Experiment 1: report the latest run (or RUN_ID=...)
	uv run jevbench intent report $${RUN_ID:+--run-id "$$RUN_ID"}

intent-sync: ## Experiment 1: mirror the latest run (or RUN_ID=...) into Phoenix
	uv run jevbench intent phoenix-sync $${RUN_ID:+--run-id "$$RUN_ID"}

rerank-grocery-candidates: ## Experiment 2: write keyword-matched searches to data/rerank/grocery_review.yaml for review
	uv run jevbench rerank prepare --grocery-candidates

rerank-prepare: ## Experiment 2: build data/rerank/splits/*.jsonl from the ESCI files (needs the grocery review)
	uv run jevbench rerank prepare

rerank-check: ## Experiment 2: one live search per system, to validate credentials and request shapes
	uv run jevbench rerank check --systems $(RERANK_SYSTEMS)

rerank-smoke: ## Experiment 2: 5 searches x every system on the practice splits, then report
	scripts/run_matrix.sh rerank "smoke-$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}" "practice practice-nomatch" "$(RERANK_SYSTEMS)" "1" \
		--limit 5 --warmup 2

rerank-practice: ## Experiment 2: tuning run over the practice splits (scratch: allowed from uncommitted code)
	scripts/run_matrix.sh rerank "practice-$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}" "practice practice-nomatch" "$(RERANK_SYSTEMS)" "1" \
		--allow-dirty

rerank-tune: ## Experiment 2: pick abstention thresholds from a practice run (RUN_ID=...)
	uv run jevbench rerank tune-thresholds --run-id "$$RUN_ID"

rerank-run: ## Experiment 2: every system x main, nomatch, reversed, and 3 consistency repeats (needs committed code)
	run_id="$${RUN_ID:-$$(date -u +%Y%m%dT%H%M%SZ)}"; \
	scripts/run_matrix.sh rerank "$$run_id" "main nomatch reversed" "$(RERANK_SYSTEMS)" "1" && \
	scripts/run_matrix.sh rerank "$$run_id" "consistency" "$(RERANK_SYSTEMS)" "1 2 3"

rerank-report: ## Experiment 2: report the latest run (or RUN_ID=...)
	uv run jevbench rerank report $${RUN_ID:+--run-id "$$RUN_ID"}

audit: ## Before publishing: scan every file git would commit for the gateway host, API keys, and .audit-denylist
	uv run jevbench audit

test: ## Offline unit tests
	uv run pytest -q

coverage: ## Offline unit tests with a coverage report (fails under 85%)
	uv run pytest -q --cov --cov-report=term-missing

lint: ## Ruff lint + format check, YAML lint
	uv run ruff check src tests && uv run ruff format --check src tests
	uv run yamllint -c .yamllint.yaml data/intent/intents.yaml data/rerank docker-compose.yml .yamllint.yaml \
		.github/workflows

typecheck: ## Pyright
	uv run pyright

validate: lint typecheck coverage ## Everything CI runs
