#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_arrays_pro.py  (with fuzzy prefix matching for directory compare)

Adds:
- Fuzzy pairing in directory comparison: files like
  last_hidden_state_onnx.npy  <->  last_hidden_state_torch.npy
  will be auto-matched by common prefix before the last underscore/hyphen token.
- Toggle with --strict (disables fuzzy pairing). Default: enabled.

Existing features:
- .npy/.npz/.bin(float32)/.txt (one float per line)
- File or directory comparison
- Automatic axis search and vectorized Top-K ranking
- Auto-watching (poll) for files/dirs
- Rich metrics (MSE/MAE/Max/Cos/SNR) and optional element preview
"""
import argparse
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

np.set_printoptions(suppress=True, precision=8)

SUPPORTED_EXT = ('.npy', '.npz', '.bin', '.txt')

# ---------------------------
# Loading & metrics
# ---------------------------

def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    diff = a - b
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    maxdiff = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    cos = _safe_cosine(a, b)
    if mse > 0:
        signal_power = float(np.mean(a * a))
        snr = float(10.0 * np.log10(signal_power / mse)) if mse > 0 else float('inf')
    else:
        snr = float('inf')
    return {"mse": mse, "mae": mae, "max": maxdiff, "cos": cos, "snr": snr}

def _can_broadcast(from_shape: Tuple[int, ...], to_shape: Tuple[int, ...]) -> bool:
    f = list(from_shape)[::-1]
    t = list(to_shape)[::-1]
    for i in range(max(len(f), len(t))):
        fd = f[i] if i < len(f) else 1
        td = t[i] if i < len(t) else 1
        if not (fd == td or fd == 1):
            return False
    return True

def _broadcast_to(x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    try:
        return np.broadcast_to(x, shape)
    except ValueError:
        if _can_broadcast(x.shape, shape):
            pad = len(shape) - x.ndim
            if pad > 0:
                x = x.reshape((1,) * pad + x.shape)
            return np.broadcast_to(x, shape)
        raise

def load_array(path: Union[str, Path], reference_shape: Optional[Tuple[int, ...]] = None) -> Union[np.ndarray, np.lib.npyio.NpzFile]:
    p = str(path)
    if p.endswith('.npy'):
        return np.load(p)
    elif p.endswith('.npz'):
        return np.load(p)
    elif p.endswith('.bin'):
        arr = np.fromfile(p, dtype=np.float32)
        if reference_shape is not None and int(np.prod(reference_shape)) == arr.size:
            try:
                arr = arr.reshape(reference_shape)
            except Exception:
                pass
        return arr
    elif p.endswith('.txt'):
        try:
            arr = np.loadtxt(p, dtype=np.float32, ndmin=1)
            return arr
        except Exception:
            arr = np.genfromtxt(p, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr
    else:
        raise ValueError(f"Unsupported file format: {p}")

# ---------------------------
# Axis search
# ---------------------------

def _axis_search_best_slices(big: np.ndarray,
                             small: np.ndarray,
                             topk: int = 5,
                             rank_metric: str = "cos") -> List[Tuple[int, int, Dict[str, float]]]:
    results: List[Tuple[int, int, Dict[str, float]]] = []

    small = small.astype(np.float32, copy=False)
    if np.isnan(small).any():
        small = np.nan_to_num(small, copy=False)

    for axis in range(big.ndim):
        target_shape = big.shape[:axis] + big.shape[axis+1:]
        if not _can_broadcast(small.shape, target_shape):
            continue

        try:
            small_b = _broadcast_to(small, target_shape).astype(np.float32, copy=False)
        except Exception:
            continue

        A = big.shape[axis]
        perm = (axis,) + tuple(i for i in range(big.ndim) if i != axis)
        big_perm = np.transpose(big, perm).astype(np.float32, copy=False)
        rest = int(np.prod(target_shape)) if target_shape else 1
        big_2d = big_perm.reshape(A, rest)
        small_1d = small_b.reshape(rest)

        diff = big_2d - small_1d
        mse = np.mean(diff * diff, axis=1)
        mae = np.mean(np.abs(diff), axis=1)
        maxdiff = np.max(np.abs(diff), axis=1) if rest > 0 else np.zeros(A, dtype=np.float32)

        vnorm = float(np.linalg.norm(small_1d))
        row_norms = np.linalg.norm(big_2d, axis=1)
        dots = big_2d @ small_1d
        cos = np.zeros(A, dtype=np.float32)
        if vnorm == 0:
            cos = (row_norms == 0).astype(np.float32)
        else:
            nz = row_norms != 0
            cos[nz] = (dots[nz] / (row_norms[nz] * vnorm)).astype(np.float32)
            cos[~nz] = 0.0

        signal_power = float(np.mean(small_1d * small_1d))
        with np.errstate(divide='ignore'):
            snr = np.where(mse > 0, 10.0 * np.log10(signal_power / mse), np.inf).astype(np.float32)

        if rank_metric == "cos":
            order = np.argsort(-cos)
        elif rank_metric == "mse":
            order = np.argsort(mse)
        elif rank_metric == "mae":
            order = np.argsort(mae)
        else:
            order = np.argsort(-cos)

        k = min(topk, A)
        for ii in order[:k]:
            metrics = {
                "mse": float(mse[ii]),
                "mae": float(mae[ii]),
                "max": float(maxdiff[ii]),
                "cos": float(cos[ii]),
                "snr": float(snr[ii]) if np.isfinite(snr[ii]) else float('inf'),
            }
            results.append((axis, int(ii), metrics))

    if len(results) > 1:
        if rank_metric == "cos":
            results.sort(key=lambda x: -x[2]["cos"])
        elif rank_metric == "mse":
            results.sort(key=lambda x: x[2]["mse"])
        elif rank_metric == "mae":
            results.sort(key=lambda x: x[2]["mae"])
        else:
            results.sort(key=lambda x: -x[2]["cos"])

    return results[:topk]

# ---------------------------
# Core comparison
# ---------------------------

def compare_arrays(a: np.ndarray,
                   b: np.ndarray,
                   topk: int = 5,
                   rank_metric: str = "cos",
                   force_axis_search: bool = False,
                   print_elements: bool = False,
                   label_a: str = "A",
                   label_b: str = "B") -> None:
    same_shape = a.shape == b.shape
    if same_shape and not force_axis_search:
        print(f"Shapes match: {a.shape}")
        if np.isnan(a).any():
            print("Warning: A contains NaN")
        if np.isnan(b).any():
            print("Warning: B contains NaN")
        m = _metrics(a, b)
        print(f"  MSE: {m['mse']:.8g} | MAE: {m['mae']:.8g} | Max: {m['max']:.8g} | Cos: {m['cos']:.8g} | SNR(dB): {m['snr']:.3f}")
        if print_elements:
            flat_a = a.ravel()
            flat_b = b.ravel()
            n = flat_a.size
            h = min(50, n)
            print(f"First {h} of {label_a}: {flat_a[:h]}")
            print(f"First {h} of {label_b}: {flat_b[:h]}")
            print(f"Last {h} of {label_a}: {flat_a[-h:]}")
            print(f"Last {h} of {label_b}: {flat_b[-h:]}")
        return

    try:
        if _can_broadcast(b.shape, a.shape):
            b_b = _broadcast_to(b, a.shape)
            print(f"Broadcasted {label_b} {b.shape} -> {a.shape} to compare with {label_a}.")
            m = _metrics(a, b_b)
            print(f"  MSE: {m['mse']:.8g} | MAE: {m['mae']:.8g} | Max: {m['max']:.8g} | Cos: {m['cos']:.8g} | SNR(dB): {m['snr']:.3f}")
            return
    except Exception:
        pass

    print(f"Shapes differ: {a.shape} vs {b.shape}. Attempting automatic axis search...")
    best = _axis_search_best_slices(a, b, topk=topk, rank_metric=rank_metric)
    reverse = _axis_search_best_slices(b, a, topk=topk, rank_metric=rank_metric)

    if not best and not reverse:
        print("No compatible axis slice found for comparison (broadcasting failed).")
        return

    def _fmt_result(tag: str, res: List[Tuple[int,int,Dict[str,float]]]):
        if not res: return
        print(f"\nTop-{len(res)} matches by slicing {tag}:")
        for (ax, idx, m) in res:
            print(f"  axis={ax}, index={idx} -> MSE={m['mse']:.6g}, MAE={m['mae']:.6g}, Max={m['max']:.6g}, Cos={m['cos']:.6g}, SNR(dB)={m['snr']:.3f}")

    _fmt_result(label_a, best)
    _fmt_result(label_b, reverse)

def _npz_common_keys(nz1: np.lib.npyio.NpzFile, nz2: np.lib.npyio.NpzFile) -> List[str]:
    return [k for k in nz1.files if k in nz2.files]

def compare_files(file1: Union[str, Path],
                  file2: Union[str, Path],
                  topk: int = 5,
                  rank_metric: str = "cos",
                  print_elements: bool = False,
                  force_axis_search: bool = False,
                  common_base: Optional[str] = None) -> None:
    file1 = str(file1); file2 = str(file2)
    if common_base:
        display1 = os.path.relpath(file1, common_base)
        display2 = os.path.relpath(file2, common_base)
    else:
        display1 = file1
        display2 = file2
    print(f"\n=== Comparing ===\n{display1}\n{display2}")

    ref_shape1 = None
    ref_shape2 = None

    if file1.endswith('.npy'):
        try:
            ref_shape1 = tuple(np.load(file1).shape)
        except Exception:
            pass
    if file2.endswith('.npy'):
        try:
            ref_shape2 = tuple(np.load(file2).shape)
        except Exception:
            pass

    a = load_array(file1, reference_shape=ref_shape2)
    b = load_array(file2, reference_shape=ref_shape1)

    if isinstance(a, np.lib.npyio.NpzFile) and isinstance(b, np.lib.npyio.NpzFile):
        keys = _npz_common_keys(a, b)
        if not keys:
            print("Warning: No common keys in NPZ files.")
            return
        for k in keys:
            print(f"\n[Key: {k}]")
            compare_arrays(a[k], b[k],
                           topk=topk, rank_metric=rank_metric,
                           force_axis_search=force_axis_search,
                           print_elements=print_elements,
                           label_a="A", label_b="B")
    elif isinstance(a, np.lib.npyio.NpzFile) or isinstance(b, np.lib.npyio.NpzFile):
        print("Warning: One file is NPZ and the other is not. Cannot compare directly.")
        return
    else:
        compare_arrays(a, b,
                       topk=topk, rank_metric=rank_metric,
                       force_axis_search=force_axis_search,
                       print_elements=print_elements,
                       label_a="A", label_b="B")

# ---------------------------
# Directory matching (with fuzzy pairing)
# ---------------------------

def _iter_files(root: str, recursive: bool) -> Iterable[str]:
    """Iterate over supported files in a directory."""
    if recursive:
        for r, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(SUPPORTED_EXT):
                    yield os.path.join(r, f)
    else:
        try:
            for entry in os.scandir(root):
                if entry.is_file() and entry.name.lower().endswith(SUPPORTED_EXT):
                    yield entry.path
        except OSError as e:
            print(f"Error listing directory {root}: {e}", file=sys.stderr)

def _rel_dir_and_name(base: str, full: str) -> Tuple[str, str]:
    rel = os.path.relpath(full, base)
    rd = os.path.dirname(rel)
    name = os.path.basename(rel)
    return rd, name

def _split_tokens(stem: str) -> List[str]:
    toks = [t for t in re.split(r'[_\\-]+', stem) if t]
    return toks

def _best_prefix_match(fname: str, candidates: List[str]) -> Optional[str]:
    """
    Return best candidate filename (string) sharing the same tokens[:-1] with fname.
    Both fname and candidates are basenames within the same directory.
    """
    stem1 = Path(fname).stem
    toks1 = _split_tokens(stem1)
    if len(toks1) < 2:
        return None
    base1 = toks1[:-1]  # common prefix tokens

    ext1 = Path(fname).suffix.lower()
    best = None
    best_score = -1

    for cand in candidates:
        stem2 = Path(cand).stem
        toks2 = _split_tokens(stem2)
        if len(toks2) < 2:
            continue
        if toks2[:-1] != base1:
            continue  # must match common prefix tokens
        score = 1000  # strong match on tokens
        ext2 = Path(cand).suffix.lower()
        if ext2 == ext1:
            score += 50  # prefer same extension
        # tie-breaker: longer stems slightly preferred (more specific)
        score += min(len(stem1), len(stem2))
        if score > best_score:
            best_score = score
            best = cand
    return best

def compare_directories(dir1: Union[str, Path],
                        dir2: Union[str, Path],
                        topk: int = 5,
                        rank_metric: str = "cos",
                        print_elements: bool = False,
                        force_axis_search: bool = False,
                        strict: bool = False,
                        recursive: bool = False) -> None:
    dir1 = os.path.abspath(str(dir1))
    dir2 = os.path.abspath(str(dir2))
    common_base = os.path.commonpath([dir1, dir2])
    if os.path.isfile(common_base):
        common_base = os.path.dirname(common_base)

    # Build index for dir2 by relative directory
    index2: Dict[str, List[str]] = {}
    for f2 in _iter_files(dir2, recursive=recursive):
        rd2, name2 = _rel_dir_and_name(dir2, f2)
        index2.setdefault(rd2, []).append(name2)

    used_in_dir2: set = set()

    for f1 in _iter_files(dir1, recursive=recursive):
        rd1, name1 = _rel_dir_and_name(dir1, f1)
        candidate_exact = os.path.join(dir2, rd1, name1)

        if os.path.exists(candidate_exact):
            pair = candidate_exact
            reason = "exact"
        else:
            pair = None
            reason = ""
            if not strict:
                # fuzzy by common prefix before last token
                cand_list = index2.get(rd1, [])
                b = _best_prefix_match(name1, cand_list)
                if b is not None:
                    pair = os.path.join(dir2, rd1, b)
                    reason = "fuzzy-prefix"

        if pair and os.path.exists(pair):
            if pair in used_in_dir2:
                display_pair = os.path.relpath(pair, common_base)
                display_f1 = os.path.relpath(f1, common_base)
                print(f"Warning: {display_pair} already matched; skipping duplicate pair for {display_f1}")
                continue
            if reason == "fuzzy-prefix":
                print(f"Info: Fuzzy matched by prefix → {name1}  ~  {os.path.basename(pair)}")
            compare_files(f1, pair, topk=topk, rank_metric=rank_metric,
                          print_elements=print_elements, force_axis_search=force_axis_search,
                          common_base=common_base)
            used_in_dir2.add(pair)
        else:
            display_f1 = os.path.relpath(f1, common_base)
            display_dir2 = os.path.relpath(dir2, common_base)
            print(f"Warning: Missing counterpart for {display_f1} in {display_dir2} (strict={strict})")

# ---------------------------
# Watching / polling
# ---------------------------

def _snapshot(paths: List[str], recursive: bool) -> Dict[str, float]:
    stat: Dict[str, float] = {}
    for p in paths:
        if os.path.isdir(p):
            for f in _iter_files(p, recursive=recursive):
                try:
                    stat[f] = os.path.getmtime(f)
                except FileNotFoundError:
                    pass
        elif os.path.isfile(p):
            try:
                stat[p] = os.path.getmtime(p)
            except FileNotFoundError:
                pass
    return stat

def _resolve_pairs(path1: str, path2: str, strict: bool, recursive: bool) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if os.path.isdir(path1) and os.path.isdir(path2):
        base1 = os.path.abspath(path1)
        base2 = os.path.abspath(path2)
        # Build index for dir2
        index2: Dict[str, List[str]] = {}
        for f2 in _iter_files(base2, recursive=recursive):
            rd2, name2 = _rel_dir_and_name(base2, f2)
            index2.setdefault(rd2, []).append(name2)
        used_in_dir2: set = set()
        for f1 in _iter_files(base1, recursive=recursive):
            rd1, name1 = _rel_dir_and_name(base1, f1)
            exact = os.path.join(base2, rd1, name1)
            if os.path.exists(exact):
                if exact not in used_in_dir2:
                    pairs.append((f1, exact))
                    used_in_dir2.add(exact)
                continue
            if not strict:
                cand_list = index2.get(rd1, [])
                b = _best_prefix_match(name1, cand_list)
                if b is not None:
                    f2p = os.path.join(base2, rd1, b)
                    if f2p not in used_in_dir2:
                        pairs.append((f1, f2p))
                        used_in_dir2.add(f2p)
    elif os.path.isfile(path1) and os.path.isfile(path2):
        pairs.append((path1, path2))
    return pairs

def watch_and_compare(path1: str,
                      path2: str,
                      interval: float = 1.0,
                      topk: int = 5,
                      rank_metric: str = "cos",
                      print_elements: bool = False,
                      force_axis_search: bool = False,
                      strict: bool = False,
                      recursive: bool = False) -> None:
    print(f"[Watcher] Monitoring changes every {interval:.1f}s. Press Ctrl+C to stop.")
    common_base = os.path.commonpath([os.path.abspath(path1), os.path.abspath(path2)])
    if os.path.isfile(common_base):
        common_base = os.path.dirname(common_base)
    last = _snapshot([path1, path2], recursive=recursive)
    last_pairs = _resolve_pairs(path1, path2, strict=strict, recursive=recursive)

    # Initial run
    if os.path.isdir(path1) and os.path.isdir(path2):
        compare_directories(path1, path2, topk, rank_metric, print_elements, force_axis_search, strict, recursive=recursive)
    elif os.path.isfile(path1) and os.path.isfile(path2):
        compare_files(path1, path2, topk, rank_metric, print_elements, force_axis_search, common_base=common_base)
    else:
        print("Error: Both arguments must be either files or directories to watch.")
        return

    try:
        while True:
            time.sleep(interval)
            cur = _snapshot([path1, path2], recursive=recursive)
            if cur != last:
                pairs = _resolve_pairs(path1, path2, strict=strict, recursive=recursive)
                changed = set()
                for f, mtime in cur.items():
                    if f not in last or last.get(f) != mtime:
                        changed.add(f)
                if os.path.isdir(path1) and os.path.isdir(path2):
                    dirty = [(a, b) for (a, b) in pairs if (a in changed or b in changed)]
                    if not dirty:
                        dirty = pairs
                    print("\n[Watcher] Change detected. Re-running comparisons for impacted files...")
                    for a, b in dirty:
                        if os.path.exists(a) and os.path.exists(b):
                            compare_files(a, b, topk, rank_metric, print_elements, force_axis_search, common_base=common_base)
                else:
                    print("\n[Watcher] Change detected. Re-running comparison...")
                    compare_files(path1, path2, topk, rank_metric, print_elements, force_axis_search, common_base=common_base)
                last = cur
                last_pairs = pairs
    except KeyboardInterrupt:
        print("\n[Watcher] Stopped.")

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare tensors/arrays with smart axis search, fuzzy prefix pairing, and auto-watching.")
    parser.add_argument("path1", type=str, help="First file or directory (.npy/.npz/.bin/.txt)")
    parser.add_argument("path2", type=str, help="Second file or directory (.npy/.npz/.bin/.txt)")
    parser.add_argument("-p", "--print", action="store_true", help="Print first/last 50 elements (when shapes match)")
    parser.add_argument("--topk", type=int, default=5, help="Top-K results to show for axis search")
    parser.add_argument("--rank-metric", type=str, default="cos", choices=["cos", "mse", "mae"], help="Ranking metric for axis search")
    parser.add_argument("--force-axis", action="store_true", help="Force axis search even when shapes match")
    parser.add_argument("--watch", action="store_true", help="Enable polling watcher to auto-compare on changes")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval (seconds) when --watch is enabled")
    parser.add_argument("--strict", action="store_true", help="Disable fuzzy prefix pairing in directory compare")
    parser.add_argument("-r", "--recursive", action="store_true", help="Enable recursive search in directories. Default is non-recursive.")

    args = parser.parse_args()

    path1 = args.path1
    path2 = args.path2

    if args.watch:
        watch_and_compare(path1, path2, interval=args.interval,
                          topk=args.topk, rank_metric=args.rank_metric,
                          print_elements=args.print, force_axis_search=args.force_axis,
                          strict=args.strict, recursive=args.recursive)
        return

    if os.path.isfile(path1) and os.path.isfile(path2):
        common_base = os.path.commonpath([os.path.abspath(path1), os.path.abspath(path2)])
        if os.path.isfile(common_base):
            common_base = os.path.dirname(common_base)
        compare_files(path1, path2, topk=args.topk, rank_metric=args.rank_metric,
                      print_elements=args.print, force_axis_search=args.force_axis,
                      common_base=common_base)
    elif os.path.isdir(path1) and os.path.isdir(path2):
        compare_directories(path1, path2, topk=args.topk, rank_metric=args.rank_metric,
                            print_elements=args.print, force_axis_search=args.force_axis,
                            strict=args.strict, recursive=args.recursive)
    else:
        print("Error: Both arguments must be either files or directories")
        sys.exit(1)

if __name__ == "__main__":
    main()
