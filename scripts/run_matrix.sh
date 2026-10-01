#!/usr/bin/env bash
# Run every system over the given splits and repeats under one run id, then write the report.
#
# Usage: scripts/run_matrix.sh RUN_ID "SPLITS" "SYSTEMS" "REPEATS" [extra jevbench run args...]
#   scripts/run_matrix.sh 20260929T120000Z "main profanity" "jev baseline" "1 2 3"
#   scripts/run_matrix.sh smoke-1 "practice" "jev baseline" "1" --limit 5 --warmup 2
set -euo pipefail

if [[ $# -lt 4 ]]; then
  sed -n '2,6p' "$0" >&2
  exit 2
fi

run_id="$1"
read -r -a splits <<<"$2"
read -r -a systems <<<"$3"
read -r -a repeats <<<"$4"
shift 4

for system in "${systems[@]}"; do
  for split in "${splits[@]}"; do
    uv run jevbench run --system "$system" --split "$split" --repeats "${repeats[@]}" --run-id "$run_id" "$@"
  done
done

uv run jevbench report --run-id "$run_id"
