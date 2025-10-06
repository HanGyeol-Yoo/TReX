#!/bin/bash
set -euo pipefail

IFS=' ' read -r -a TOKENIZER_PATHS <<< "${TREX_TOKENIZER_PATHS:-/path/to/tokenizers/fineweb2_trex/30gb_200k}"
DATA_ROOT="${TREX_VALID_DATA_ROOT:-/path/to/valid_data}"
OUTPUT_ROOT="${TREX_RECORD_ROOT:-/path/to/records}"

mkdir -p "$OUTPUT_ROOT"

for TOK in "${TOKENIZER_PATHS[@]}"; do
    echo ">>> Running for $TOK"

    TOK_NAME=$(basename "$TOK")
    OUT_PKL="${OUTPUT_ROOT}/${TOK_NAME}_lang_concat_results.pkl"

    python3 -m trex.calculate_length \
        --ds_root "$DATA_ROOT" \
        --tok_root "$TOK" \
        --out_pkl "$OUT_PKL"

    if [ -d "$DATA_ROOT" ]; then
        find "$DATA_ROOT" -maxdepth 2 -name "cache*" -type d -exec rm -rf {} +
    fi
done
