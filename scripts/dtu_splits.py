#!/usr/bin/env python
"""Generate nested train/val/test splits for the DTU correlation study.

For a prepared scene (one transforms.json with all views, plus images/ and
masks/), this writes one experiment directory per (N, seed):

    <out>/<scene>_n<N>_s<seed>/
        transforms.json          same cameras, with the three split lists
        images  -> symlink to the shared prepared images
        masks   -> symlink
        scale_mat.npy -> symlink

Design:
    test set   fixed per scene, evenly spaced over the view indices, never
               trained, identical across every N and seed.
    train set  drawn from the remaining pool. For each seed a single random
               permutation is taken, and the N=5,10,20,30,40 subsets are the
               nested prefixes of it, so N=5 is contained in N=10 and so on.
    val set    equal to the test set, used only for during-training logging.

Nerfstudio reads train_filenames/val_filenames/test_filenames from
transforms.json and these override eval_mode. The lists hold file_path
strings exactly as they appear in frames.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

N_LIST = (5, 10, 20, 30, 40)
SEEDS = (0, 1, 2)
N_TEST = 6


def test_indices(n_views: int, n_test: int) -> list[int]:
    return sorted({round(i * (n_views - 1) / (n_test - 1)) for i in range(n_test)})


def relink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def generate(prepared: Path, out_root: Path, scene: str, *,
             n_list=N_LIST, seeds=SEEDS, n_test=N_TEST) -> list[dict]:
    base = json.loads((prepared / "transforms.json").read_text())
    frames = base["frames"]
    nv = len(frames)
    paths = [f["file_path"] for f in frames]

    test_idx = test_indices(nv, n_test)
    pool = [i for i in range(nv) if i not in test_idx]
    if max(n_list) > len(pool):
        raise ValueError(f"{scene}: N={max(n_list)} exceeds pool {len(pool)}")
    test_files = [paths[i] for i in test_idx]

    rows = []
    for seed in seeds:
        perm = [int(i) for i in np.random.default_rng(seed).permutation(pool)]
        for n in n_list:
            train_files = [paths[i] for i in perm[:n]]
            exp = out_root / f"{scene}_n{n:02d}_s{seed}"
            exp.mkdir(parents=True, exist_ok=True)

            doc = dict(base)
            doc["train_filenames"] = train_files
            doc["val_filenames"] = test_files
            doc["test_filenames"] = test_files
            (exp / "transforms.json").write_text(json.dumps(doc, indent=2))

            relink(exp / "images", prepared / "images")
            relink(exp / "masks", prepared / "masks")
            if (prepared / "scale_mat.npy").exists():
                relink(exp / "scale_mat.npy", prepared / "scale_mat.npy")

            rows.append({"scene": scene, "n": n, "seed": seed,
                         "n_train": len(train_files), "n_test": len(test_files),
                         "dir": str(exp)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared", type=Path, help="prepared scene dir")
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--n-list", type=int, nargs="+", default=list(N_LIST))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    scene = args.scene or args.prepared.name
    rows = generate(args.prepared, args.out_root, scene,
                    n_list=tuple(args.n_list), seeds=tuple(args.seeds))
    for r in rows:
        print(f"  {r['dir']}: {r['n_train']} train, {r['n_test']} test")
    print(f"\n{len(rows)} experiment dirs for {scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
