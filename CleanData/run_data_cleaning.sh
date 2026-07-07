#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_DIR="${RAW_DIR:-data-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-data-v2/processed}"

python src/data_cleaning.py \
  --raw-dir "$RAW_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
