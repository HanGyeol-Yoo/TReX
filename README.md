# TReX Tokenizer Regression for Optimal Data Mixture

This repository contains utilities and experiments for exploring tokenizer design and
mixture optimization for language model pre-training. It includes:

- Scripts for training and post-processing multiple tokenizers.
- Utilities for generating mixture configurations using Dirichlet sampling.
- A bytes-per-token analysis pipeline used to regress tokenization efficiency across datasets.
- Research notebooks used to check rank invariance and to fit the regression model that
  suggests optimal data mixtures for downstream large-scale training.

## Repository Structure

| Path | Description |
|------|-------------|
| `train/` | CLI utilities for training a tokenizer and preparing the associated training corpora. |
| `mixture/` | Tools for sampling domain mixtures and emitting YAML configs for regression experiments. |
| `notebook/` | Jupyter notebooks that document exploratory data analysis and model fitting. |
| `calculate_length.py` | Computes token length statistics for every tokenizer/dataset pair. |
| `make_config.sh` | Shell helper that wraps `mixture/make_config.py`. |
| `train_various_tokenizer.sh` | Batch utility that runs the tokenizer training pipeline across many configs. |

## Prerequisites

This project targets Python 3.10+ and depends on common machine learning libraries. To get
started, create a virtual environment and install the requirements manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install click datasets filelock numpy omegaconf pandas tqdm transformers pyyaml
```

Some workflows (for example, evaluating tokenizers) require access to Hugging Face datasets and
may need environment authentication. Set the appropriate `HF_HOME` or `HUGGINGFACE_HUB_CACHE`
variables if you store datasets outside the repository.

## Training Tokenizers

The tokenizer training pipeline lives in `train/train.py` and expects:

- A YAML configuration file that specifies the relative weighting of each domain under the
  `train` key.
- Text corpora stored in `train/<domain_name>` directories.

To launch a single experiment manually:

```bash
python -m trex.train.train \
  --output_dir ./tokenizers/example_tokenizer \
  --cfg_file ./configs/domain_mix.yaml \
  --num_bytes 1073741824 \
  --vocab_size 64000 \
  --setting 0
```

After training, run post-processing to build the Hugging Face tokenizer artifact:

```bash
python -m trex.train.preprocess \
  --process_tgt ./tokenizers/example_tokenizer \
  --output_dir ./tokenizers/example_tokenizer/post_processed
```

The `train_various_tokenizer.sh` script automates this workflow across multiple configurations.
It relies on environment variables so you can point to your own config, output, and data roots:

```bash
export TREX_CONFIG_DIR=/absolute/path/to/configs
export TREX_TOKENIZER_OUTPUT_ROOT=/absolute/path/to/tokenizers
export TREX_DATA_ROOT=/absolute/path/to/raw_corpora
./train_various_tokenizer.sh
```

## Generating Mixture Configurations

Use `mixture/make_config.py` to sample domain weightings for regression experiments. The script
implements a temperature-controlled Dirichlet sampling strategy with clipping bounds derived from
empirical token usage. Generate 512 candidate mixtures into `config_1m/` with:

```bash
python mixture/make_config.py --output_folder config_1m --num_configs 512
```

Each output file is a YAML configuration containing both train and validation mixtures, along with
metadata documenting the sampling hyperparameters and proxy model settings.

## Calculating Token Length Statistics

`calculate_length.py` computes the tokenization length distribution for every combination of
trained tokenizer and dataset. Provide the tokenizer root directory, dataset root, and output path:

```bash
python calculate_length.py \
  --tok_root /absolute/path/to/tokenizers \
  --ds_root /absolute/path/to/datasets \
  --out_pkl ./artifacts/token_lengths.pkl
```

Key environment variables:

- `TREX_DEFAULT_DS_ROOT` – optional override for the default dataset root (`./datasets`).
- `NUM_PROCS_PER_TOKENIZER` and `MAX_WORKERS` (constants in the script) can be adjusted if you need
  to tune concurrency for your hardware.

The resulting pickle file contains a table where each row corresponds to a tokenizer and each column
records the token lengths for a dataset. These statistics drive the regression notebook that
estimates bytes-per-token efficiency across mixtures.

## Notebooks

The `notebook/` directory contains two primary notebooks:

1. **`trex_regression_model_train_infer.ipynb`** – trains the regression model and predicts optimal
   mixture weights for new tokenizer candidates.
2. **`check_rank_invariance.ipynb`** – validates that the regression preserves rank ordering across
   evaluation splits.

Both notebooks now use anonymized, relative paths. Open them in Jupyter Lab or VS Code after
activating your virtual environment.

## Reproducing Experiments

A typical workflow to reproduce the paper results is:

1. Generate candidate mixtures (`mixture/make_config.py`).
2. Train tokenizers for each mixture (`train_various_tokenizer.sh`).
3. Compute bytes-per-token statistics (`calculate_length.py`).
4. Analyze and select the best mixtures in the regression notebook.

Make sure to cache datasets locally and adjust the script arguments to match your storage layout.

## License

This repository inherits the licensing terms of the original TReX project. If you plan to redistribute
artifacts, consult the upstream license and ensure compliance with the datasets you consume.

