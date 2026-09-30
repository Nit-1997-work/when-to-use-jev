#!/usr/bin/env bash
# Wait until Phoenix answers its health check, then register Jev's token price.
# Usage: scripts/phoenix_up.sh [base_url]   (default: PHOENIX_BASE_URL from the environment or .env, else :6006)
set -euo pipefail

if [[ -z "${PHOENIX_BASE_URL:-}" && -f .env ]]; then
  # Same value jevbench reads: the last PHOENIX_BASE_URL line in .env, without surrounding quotes.
  PHOENIX_BASE_URL="$(sed -n -E "s/^PHOENIX_BASE_URL=['\"]?([^'\"]*)['\"]?[[:space:]]*$/\1/p" .env | tail -n 1)"
fi
base_url="${1:-${PHOENIX_BASE_URL:-http://localhost:6006}}"

for _ in {1..60}; do
  if curl -sf "${base_url}/healthz" >/dev/null; then
    uv run jevbench phoenix-setup
    exit 0
  fi
  sleep 1
done

echo "Phoenix did not become healthy at ${base_url} within 60s." >&2
exit 1
