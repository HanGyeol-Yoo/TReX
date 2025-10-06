TOK_PATH=(
    /home/work/mlp/trex/tokenizers/fineweb2_trex/30gb_200k
)

for TOK in "${TOK_PATH[@]}"
do
    echo ">>> Running for $TOK"
    
    REL_PATH="${TOK#*/tokenizers/}"

    python3 -m trex.calculate_length \
        --ds_root /home/work/mlp/trex/dataFW/valid_txt_ratio_hf \
        --tok_root $TOK \
        --out_pkl "/home/work/mlp/trex/records/comp/fineweb2_trex/$REL_PATH/lang_concat_results.pkl"
    rm -rf /home/work/mlp/trex/dataFW/valid_txt_ratio_hf/*/cache*
done