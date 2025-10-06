#!/bin/bash
set -euo pipefail

OUTPUT_FOLDER="${TREX_CONFIG_OUTPUT_DIR:-/path/to/configs}"
NUM_CONFIGS="${TREX_NUM_CONFIGS:-64}"

python3 -m trex.mixture.make_config \
    --output_folder "${OUTPUT_FOLDER}" \
    --num_configs "${NUM_CONFIGS}"
