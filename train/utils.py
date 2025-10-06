import os
import random
import numpy as np
import string
from tqdm import tqdm
from pathlib import Path
from filelock import FileLock

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def utf8_trim_to_boundary(buf: bytes) -> bytes:
    if not buf:
        return buf
    i = len(buf) - 1

    while i >= 0 and (buf[i] & 0b11000000) == 0b10000000:
        i -= 1
    if i < 0:
        return b""
    start = buf[i]
    if   (start & 0b10000000) == 0:      need = 1 
    elif (start & 0b11100000) == 0b11000000:  need = 2
    elif (start & 0b11110000) == 0b11100000:  need = 3
    elif (start & 0b11111000) == 0b11110000:  need = 4
    else:
        return buf[:i]

    have = len(buf) - i
    if have < need:
        return buf[:i]
    else:
        return buf[: i + need]



def _collect_from_dir(
    data_dir: Path,
    target_bytes: int,
    loop_around: bool,
    truncated_dir: Path
):
    files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".txt") and "truncated" not in f and "split" not in f
    ]
    random.seed(41)
    random.shuffle(files)

    truncated_dir.mkdir(parents=True, exist_ok=True)

    collected = []
    byte_count = 0
    counter = 0
    bar = tqdm(total=target_bytes, desc=f"Loading from {data_dir.name}") if target_bytes > 0 else None

    while byte_count < target_bytes and files:
        fname = files[counter % len(files)]
        src_path = data_dir / fname
        fsize = os.path.getsize(src_path)

        if byte_count + fsize <= target_bytes:
            collected.append(str(src_path))
            byte_count += fsize
            if bar: bar.update(fsize)
        else:
            want = target_bytes - byte_count
            rand_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
            trunc_name = f"{src_path.stem}_truncated_{rand_suffix}{src_path.suffix}"
            trunc_path = truncated_dir / trunc_name

            with open(src_path, 'rb') as fin:
                raw = fin.read(want + 4)
            safe = utf8_trim_to_boundary(raw)

            with open(trunc_path, 'wb') as fout:
                fout.write(safe)

            real_size = len(safe)
            collected.append(str(trunc_path))
            byte_count += real_size
            if bar: bar.update(real_size)

        counter += 1
        if not loop_around and counter >= len(files):
            break

    if bar: bar.close()
    return collected, byte_count

def get_files_with_num_bytes(
    corpus_dirs: list,
    dir_ratio: list,
    num_bytes: int | None = None,
    loop_around: bool = True,
):
    dirs = [Path(_dir) for _dir in corpus_dirs]

    if len(dir_ratio) != len(dirs):
        raise ValueError(f"dir_ratio len should be {len(dirs)}. (now {len(dir_ratio)})")

    trunc_dirs = [Path(f"{d}_truncated") for d in dirs]

    if num_bytes is None:
        files_all, bytes_all = [], 0
        for d, t in zip(dirs, trunc_dirs):
            flist, bcnt = _collect_from_dir(d, float("inf"), False, t)
            files_all.extend(flist)
            bytes_all += bcnt
        print(f"Using all files: {len(files_all)} files, {bytes_all} bytes")
        return files_all, bytes_all

    targets = [int(num_bytes * r) for r in dir_ratio]

    remainder = num_bytes - sum(targets)
    if remainder:
        idx_max = max(range(len(dir_ratio)), key=lambda i: dir_ratio[i])
        targets[idx_max] += remainder

    all_files, total_bytes = [], 0
    for d, t_dir, tgt in zip(dirs, trunc_dirs, targets):
        flist, bcnt = _collect_from_dir(d, tgt, loop_around, t_dir)
        all_files.extend(flist)
        total_bytes += bcnt

    return all_files, total_bytes