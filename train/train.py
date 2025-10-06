import os
import re
import time
import json
import click
import random
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from trex.train.setting import train_or_extend_tokenizer
from trex.train.utils import ensure_dir, get_files_with_num_bytes

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

@click.command()
@click.option(
    "--output_dir",
    type=str,
    help="Where to save the trained tokenizer.",
)
@click.option(
    "--num_bytes",
    type=int,
    default=None,
    help="The maximum number of bytes to use for tokenizer training.",
)
@click.option(
    "--cfg_file",
    type=str,
    default=None,
    help="The Config File used to train tokenizer.",
)
@click.option(
    "--vocab_size",
    type=int,
    default=100000,
    help="The number of tokens in the vocabulary.",
)
@click.option(
    "--setting",
    type=int,
    default=0,
    help="Tokenizer Settings.",
)
def main(
    output_dir: str,
    cfg_file: int,
    vocab_size: int,
    setting: int,
    num_bytes: int,
):
    cfg = OmegaConf.load(cfg_file)
    corpus_dirs = []
    dir_ratio = []

    for k, v in cfg.train.items():
        corpus_dirs.append(f"./train/{k}")
        dir_ratio.append(v)

    assert len(corpus_dirs) == len(dir_ratio), "You must set len(corpus_dirs) == len(dir_ratio)"

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    print(f"We are training a tokenizer for {output_dir}", flush=True)

    os.chdir(output_dir)

    if os.path.exists("meta.json"):
        meta = json.load(open("meta.json"))
        train_files, actual_num_bytes = meta["train_files"], meta["total_bytes"]

    else:
        train_files, actual_num_bytes = get_files_with_num_bytes(corpus_dirs, dir_ratio, num_bytes)

        with open("meta.json", "w") as fo:
            meta = {}
            meta["total_bytes"] = actual_num_bytes
            meta["train_files"] = train_files
            if os.path.exists("merges.txt"):
                os.system("cp merges.txt initial_merges.txt")
                meta["num_initial_merges"] = (
                    sum(1 for line in open("initial_merges.txt")) - 1
                )
            json.dump(meta, fo, indent=5)

    start_time = time.time()

    print("Training with HF tokenizers...")
    tokenizer = train_or_extend_tokenizer(
        train_files,
        vocab_size=vocab_size,
        setting=setting
    )
    tokenizer.model.save(".")
    tokenizer.save("tokenizer.json")

    train_time = time.time() - start_time

    print(f"Train time: {train_time}", flush=True)
    print("Tokenizer info saved to " + str(output_dir), flush=True)

    with open(output_dir / "time.txt", "w") as f:
        f.write(f"{train_time:.2f}\n")


if __name__ == "__main__":
    main()
