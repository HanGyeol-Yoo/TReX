import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import concurrent.futures  

import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tqdm import tqdm

ADD_SPECIAL_TOKENS = False
TEXT_COL_CANDIDATES = ["text", "content", "raw", "document", "input", "inputs"]

NUM_PROCS_PER_TOKENIZER = 4
MAX_WORKERS = 16

DEFAULT_DS_ROOT = Path(os.environ.get("TREX_DEFAULT_DS_ROOT", "./datasets"))

def find_text_column(dataset) -> str:
    """Automatically find the text column in a dataset."""
    cols = set(dataset.column_names)
    for c in TEXT_COL_CANDIDATES:
        if c in cols:
            return c
    try:
        str_cols = [c for c in dataset.column_names
                    if dataset.features.get(c, None) and dataset.features[c].dtype == "string"]
    except Exception:
        str_cols = []
    if str_cols:
        return str_cols[0]
    raise ValueError(f"Error! candidates={TEXT_COL_CANDIDATES}, input={dataset.column_names}")

def load_tokenizer(tok_dir: Path) -> Optional[PreTrainedTokenizerFast]:
    """Load a Hugging Face format tokenizer."""
    if not tok_dir.exists(): 
        return None
    try:
        tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer: {tok_dir} ({e})")
        return None

def list_tokenizer_dirs(tok_root: Path) -> Dict[str, Path]:
    """Scan tokenizer directories with 'cfg_*' format."""
    if not tok_root.is_dir():
        raise FileNotFoundError(f"Tokenizer root directory not found: {tok_root}")
    out = {}
    for d in sorted(tok_root.glob("*")):
        name = d.name.replace("cfg_", "")
        pp = d / "post_processed"
        out[name] = pp if pp.exists() else d
    return out

def list_dataset_dirs(ds_root: Path) -> Dict[str, Path]:
    """Scan dataset directories."""
    if not ds_root.is_dir():
        raise FileNotFoundError(f"Dataset root directory not found: {ds_root}")
    out = {}
    for d in sorted(ds_root.iterdir()):
        if d.is_dir():
            out[d.name] = d
    return out

# ======================
# Core computation: average bytes per token
# ======================
def lengths_for_dataset(ds_path: Path, tok: PreTrainedTokenizerFast) -> float:
    """Compute average bytes per token for a single dataset."""
    ds = load_from_disk(str(ds_path))
    if isinstance(ds, DatasetDict) and "train" in ds:
        ds = ds["train"]

    text_col = find_text_column(ds)

    def _lengths(batch):
        texts = batch[text_col]
        safe_texts = [str(t) if t is not None else "" for t in texts]
        enc = tok(safe_texts, add_special_tokens=ADD_SPECIAL_TOKENS, truncation=False, padding=False)
        tok_lengths = []
        for t, ids in zip(safe_texts, enc["input_ids"]):
            tok_len = len(ids)
            tok_lengths.append(tok_len)
        return {"length": tok_lengths}

    ds_length = ds.map(
        _lengths,
        batched=True,
        batch_size=512,
        # --- Changed: fixed num_proc to 16 ---
        num_proc=NUM_PROCS_PER_TOKENIZER,
        # -------------------------------------
        desc=f"Tokenizing {ds_path.name} with {tok.name_or_path.split('/')[-1]}"
    )

    length = np.array(ds_length[:]['length'])
    return length

# --- Changed: separated worker function for single tokenizer ---
def process_single_tokenizer(tok_name: str, tok_path: Path, ds_dirs: Dict[str, Path]) -> Dict:
    """Perform analysis for all datasets using a single tokenizer."""
    print(f"--- Processing Tokenizer: {tok_name} ---")
    tok = load_tokenizer(tok_path)
    if tok is None:
        print(f"[SKIP] Failed to load tokenizer: {tok_name} -> {tok_path}")
        # Fill with NaN to keep consistent result structure
        row = {"tok_name": tok_name}
        for ds_name in ds_dirs.keys():
            row[ds_name] = float("nan")
        return row

    row = {"tok_name": tok_name}
    for ds_name, ds_path in ds_dirs.items():
        try:
            lengths = lengths_for_dataset(ds_path, tok)
            row[ds_name] = lengths
        except Exception as e:
            print(f"[WARN] {tok_name} / {ds_name} failed: {e}")
            row[ds_name] = float("nan")
    return row
# --------------------------------------------------------

# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser(description="Compute Bytes per Token for each tokenizer and dataset in parallel.")
    parser.add_argument("--tok_root", type=Path, required=True, help="Root directory containing the tokenizers to analyze.")
    parser.add_argument("--ds_root", type=Path, default=DEFAULT_DS_ROOT, help=f"Root directory for datasets. Default: {DEFAULT_DS_ROOT}")
    parser.add_argument("--out_pkl", type=Path, required=True, help="Path to save the resulting PKL file.")
    args = parser.parse_args()

    tok_dirs = list_tokenizer_dirs(args.tok_root)
    ds_dirs  = list_dataset_dirs(args.ds_root)

    if not tok_dirs: 
        raise RuntimeError(f"No tokenizers found: {args.tok_root}")
    if not ds_dirs: 
        raise RuntimeError(f"No datasets found: {args.ds_root}")

    rows = []
    
    # --- Changed: use ProcessPoolExecutor for parallel processing ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit each tokenizer job to the pool
        futures = {
            executor.submit(process_single_tokenizer, name, path, ds_dirs): name
            for name, path in tok_dirs.items()
        }
        
        # Collect results as each job completes (show overall progress with tqdm)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tok_dirs), desc="Overall Progress"):
            try:
                result_row = future.result()
                rows.append(result_row)
            except Exception as e:
                tok_name_failed = futures[future]
                print(f"[ERROR] Worker for tokenizer '{tok_name_failed}' failed: {e}")

    if not rows:
        print("No results processed. Exiting program.")
        return

    col_names = ["tok_name"] + list(ds_dirs.keys())
    df = pd.DataFrame(rows)
    df = df[col_names]  # Fix column order
    df.sort_values("tok_name", inplace=True)
    
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.out_pkl)
    print(f"\n✅ Saved successfully: {args.out_pkl}")

if __name__ == "__main__":
    main()
