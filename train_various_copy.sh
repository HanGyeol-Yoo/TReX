#!/bin/bash
set -euo pipefail

SETTING="${TREX_TOKENIZER_SETTING:-0}"
CONFIG_DIR="${TREX_CONFIG_DIR:-/path/to/configs}"
OUTPUT_ROOT="${TREX_TOKENIZER_OUTPUT_ROOT:-/path/to/tokenizers}"
DATA_ROOT="${TREX_DATA_ROOT:-/path/to/data}"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "❌ Config directory not found: $CONFIG_DIR" >&2
    exit 1
fi

run_train () {
    VOCAB_SIZE="$1"
    DATA_BYTES="$2"
    OUT_NAME="$3"

    for CFG_FILE in "$CONFIG_DIR"/*.yaml; do
        [ -e "$CFG_FILE" ] || continue
        BASENAME=$(basename "$CFG_FILE" .yaml)

        echo "🚀 Starting training with config: ${CFG_FILE}"

        OUTPUT_DIR="${OUTPUT_ROOT}/${OUT_NAME}/${BASENAME}"

        if [ -d "$OUTPUT_DIR" ]; then
            echo "⚠️  ${OUTPUT_DIR} already exists. Skipping train/preprocess."
            continue
        fi

        python3 -m trex.train.train \
            --output_dir "$OUTPUT_DIR" \
            --cfg_file "$CFG_FILE" \
            --num_bytes "$DATA_BYTES" \
            --vocab_size "$VOCAB_SIZE" \
            --setting "$SETTING"

        python3 -m trex.train.preprocess \
            --process_tgt "$OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR/post_processed"

        if [ -d "$DATA_ROOT" ]; then
            find "$DATA_ROOT" -name "*_trunc*" -type d -exec rm -rf {} +
        fi

        sleep 3
    done
}

# Examples:
# run_train  64000 1073741824   example_1gb_64k
# run_train  64000 5368709120   example_5gb_64k
# run_train 100000 10737418240  example_10gb_100k
run_train 200000 32212254720 example_30gb_200k
