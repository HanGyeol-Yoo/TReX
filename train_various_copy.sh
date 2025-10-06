#!/bin/bash
SETTING=0
DIR_PATH="/home/work/mlp/trex/configs/fineweb2_base_jeabal"
run_train () {
    VOCAB_SIZE=$1
    DATA_GB=$2
    OUT_NAME=$3

    for CFG_FILE in "$DIR_PATH"/*; do
        BASENAME=$(basename "$CFG_FILE" .yaml)
        CONFIG_NUM=${BASENAME#n}

        echo "🚀 Starting training with config: ${CFG_FILE}"

        OUTPUT_DIR=/home/work/mlp/trex/tokenizers/fineweb2_hq_valid/${OUT_NAME}/${BASENAME}

        # If OUTPUT_DIR exists, skip both train and preprocess
        if [ -d "$OUTPUT_DIR" ]; then
            echo "⚠️  ${OUTPUT_DIR} already exists. Skipping train/preprocess."
            continue
        fi

        python3 -m trex.train.train \
            --output_dir $OUTPUT_DIR \
            --cfg_file $CFG_FILE \
            --num_bytes $DATA_GB \
            --vocab_size $VOCAB_SIZE \
            --setting $SETTING

        wait

        python3 -m trex.train.preprocess \
            --process_tgt $OUTPUT_DIR \
            --output_dir $OUTPUT_DIR/post_processed

        # rm -rf /home/work/mlp/trex/dataFW/train/*_trunc*
        wait
        sleep 3
    done
}
# 표에 맞는 실행 목록
# run_train  64000 1073741856  1gb_64k
# run_train  64000 5368709132  5gb_64k
# run_train  100000 10737418272  10gb_100k
run_train  200000 32212254844  30gb_200k