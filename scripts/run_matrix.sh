#!/usr/bin/env bash
# Run every system over the given splits and repeats under one run id, then write the report.
#
# Usage: scripts/run_matrix.sh EXPERIMENT RUN_ID "SPLITS" "SYSTEMS" "REPEATS" [extra jevbench run args...]
#   scripts/run_matrix.sh intent 20260929T120000Z "main profanity" "jev baseline" "1"
#   scripts/run_matrix.sh rerank smoke-1 "practice" "jev-score gemini-listwise" "1" --limit 5 --warmup 2
set -euo pipefail

if [[ $# -lt 5 ]]; then
  sed -n '2,6p' "$0" >&2
  exit 2
fi

experiment="$1"
run_id="$2"
read -r -a splits <<<"$3"
read -r -a systems <<<"$4"
read -r -a repeats <<<"$5"
shift 5

for system in "${systems[@]}"; do
  for split in "${splits[@]}"; do
    uv run jevbench "$experiment" run --system "$system" --split "$split" --repeats "${repeats[@]}" \
      --run-id "$run_id" "$@"
  done
done

uv run jevbench "$experiment" report --run-id "$run_id"
