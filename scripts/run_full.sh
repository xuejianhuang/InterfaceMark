#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/sd15_terminal_smoke.yaml}"
OUTPUT="$(
  python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["output"])' \
    "$CONFIG"
)"

if [[ ! -f "$OUTPUT/PREPARE_COMPLETE" ]]; then
  interfacemark-prepare --config "$CONFIG"
fi
interfacemark-generate --input "$OUTPUT" --split calibration
interfacemark-generate --input "$OUTPUT" --split test
interfacemark-evaluate --input "$OUTPUT"

printf 'InterfaceMark run complete: %s\n' "$OUTPUT"
