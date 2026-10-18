#!/usr/bin/env python
"""Resumable batch runner for the DTU correlation study (Part A).

For every experiment directory data/dtu/experiments/<scene>_n<N>_s<seed>/ this
runs, in order: train (fixed iterations) -> render-eval -> poisson mesh export
-> mesh statistics, and appends one row to results/metrics.csv.

Robustness, as the project requires on a throttling laptop:
  * completed runs (a CSV row with status ok) are skipped, so the series can
    be stopped with Ctrl+C and resumed by re-running this script
  * every stage is isolated: a failure writes a row with the failing stage as
    status and moves on, it never aborts the other runs
  * CUDA is checked before each run; if the GPU is gone the series stops
    cleanly so nothing trains on CPU or crashes in a loop
  * warnings are silenced in the child processes to keep the logs readable

Geometry metrics and foreground-only render metrics are a later stage that
needs the official DTU evaluation data. The CSV already has the columns, left
empty here, so that stage only fills them in.

Usage:
    python scripts/run_experiments.py                 # all experiments
    python scripts/run_experiments.py --scenes scan24 # one scene
    python scripts/run_experiments.py --retry-failed  # also redo failed rows
    python scripts/run_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

STEPS = 12000
EXP_ROOT = Path("data/dtu/experiments")
OUT_ROOT = Path("outputs/exp")
RESULTS = Path("results")
CSV_PATH = RESULTS / "metrics.csv"
MESH_DIR = RESULTS / "meshes"
LOG_DIR = RESULTS / "logs"

FIELDS = [
    "exp", "scene", "n", "seed", "status",
    "psnr", "ssim", "lpips",
    "n_faces", "n_vertices", "watertight", "n_components", "n_nonmanifold",
    "chamfer", "f_score", "psnr_fg", "ssim_fg", "lpips_fg",
    "train_s", "total_s",
    "git", "nerfstudio", "torch", "steps", "timestamp",
]

ENV = {**os.environ, "PYTHONWARNINGS": "ignore::UserWarning,ignore::FutureWarning"}


def parse_exp(name: str):
    m = re.match(r"(scan\d+)_n(\d+)_s(\d+)$", name)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def versions() -> dict:
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "?"
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = "?"
    return {"git": git, "nerfstudio": _v("nerfstudio"), "torch": _v("torch")}


def cuda_ok() -> bool:
    r = subprocess.run(
        [sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
        env=ENV, capture_output=True)
    return r.returncode == 0


def done_set(retry_failed: bool) -> set[str]:
    if not CSV_PATH.exists():
        return set()
    done = set()
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok" or (not retry_failed and row["status"] != "ok"):
                done.add(row["exp"])
    return done


def append_row(row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    full = {k: row.get(k, "") for k in FIELDS}
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(full)


HELPER = Path(__file__).resolve().parent / "meshstat_helper.py"


def mesh_stats(path: Path, log) -> dict:
    """Mesh stats in a memory-capped subprocess. Returns {} on any failure, so
    a pathological mesh can never take the runner down. The .ply stays on disk
    for the later geometry stage regardless."""
    try:
        r = subprocess.run([sys.executable, str(HELPER), str(path)],
                           env=ENV, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            return json.loads(r.stdout)
        log.write(f"\nmesh_stats failed (rc {r.returncode}):\n{r.stderr[-500:]}\n")
    except Exception as e:
        log.write(f"\nmesh_stats error: {e}\n")
    return {}


def run_cmd(cmd: list[str], log) -> None:
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    r = subprocess.run(cmd, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[0]} {cmd[1]}")


def latest_config(exp: str) -> Path:
    runs = sorted((OUT_ROOT / exp / "nerfacto").glob("*/config.yml"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no config.yml for {exp}")
    return runs[-1]


def run_one(exp_dir: Path, ver: dict) -> dict:
    exp = exp_dir.name
    scene, n, seed = parse_exp(exp)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    row = {"exp": exp, "scene": scene, "n": n, "seed": seed,
           "steps": STEPS, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           **ver}
    t_start = time.time()

    with open(LOG_DIR / f"{exp}.log", "w") as log:
        try:
            t0 = time.time()
            run_cmd([
                "ns-train", "nerfacto",
                "--max-num-iterations", str(STEPS),
                "--vis", "tensorboard",
                "--output-dir", str(OUT_ROOT),
                "--experiment-name", exp,
                "--pipeline.model.predict-normals", "True",
                "--pipeline.model.background-color", "black",
                "--pipeline.model.camera-optimizer.mode", "off",
                "--pipeline.model.use-appearance-embedding", "False",
                "--pipeline.datamanager.pixel-sampler.ignore-mask", "True",
                "--data", str(exp_dir),
                "nerfstudio-data",
                "--orientation-method", "none", "--center-method", "none",
                "--auto-scale-poses", "False", "--mask-color", "0", "0", "0",
            ], log)
            row["train_s"] = round(time.time() - t0)
            cfg = latest_config(exp)
        except Exception as e:
            row["status"] = "failed:train"
            log.write(f"\nERROR {e}\n")
            row["total_s"] = round(time.time() - t_start)
            return row

        try:
            eval_json = OUT_ROOT / exp / "eval.json"
            run_cmd(["ns-eval", "--load-config", str(cfg),
                     "--output-path", str(eval_json)], log)
            r = json.load(open(eval_json))["results"]
            row.update(psnr=round(r["psnr"], 4), ssim=round(r["ssim"], 4),
                       lpips=round(r["lpips"], 4))
        except Exception as e:
            row["status"] = "failed:eval"
            log.write(f"\nERROR {e}\n")
            row["total_s"] = round(time.time() - t_start)
            return row

        try:
            mesh_out = MESH_DIR / exp
            run_cmd(["ns-export", "poisson", "--load-config", str(cfg),
                     "--output-dir", str(mesh_out),
                     "--texture-method", "point_cloud",
                     "--num-points", "1000000",
                     # Begrenzung auf die Einheitskugel, in der das Objekt per
                     # DTU Normierung sitzt. Ohne das legt Poisson eine Flaeche
                     # ueber die ferne Hintergrundpunktwolke, gemessen bis
                     # Radius 828 statt 1. Der Poisson Exporter erbt
                     # --use-bounding-box nur und beachtet es nicht, wirksam
                     # sind die OBB Parameter. obb-scale ist die volle
                     # Kantenlaenge, 2 2 2 ergibt -1 bis 1.
                     "--obb-center", "0", "0", "0",
                     "--obb-rotation", "0", "0", "0",
                     "--obb-scale", "2", "2", "2"], log)
            ply = mesh_out / "poisson_mesh.ply"
            if not ply.exists():
                raise FileNotFoundError("poisson_mesh.ply not produced")
        except Exception as e:
            row["status"] = "failed:export"
            log.write(f"\nERROR {e}\n")
            row["total_s"] = round(time.time() - t_start)
            return row

        # export succeeded, so the run counts as ok; mesh stats are a
        # non-fatal bonus and leave their columns empty if they fail
        row.update(mesh_stats(ply, log))

    row["status"] = "ok"
    row["total_s"] = round(time.time() - t_start)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--exp", default=None,
                    help="nur diesen einen Lauf, z.B. scan24_n05_s0")
    ap.add_argument("--limit", type=int, default=None,
                    help="hoechstens so viele Laeufe, fuer einen Rauchtest")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exps = sorted([d for d in EXP_ROOT.iterdir()
                   if d.is_dir() and parse_exp(d.name)],
                  key=lambda d: parse_exp(d.name))
    if args.scenes:
        exps = [d for d in exps if parse_exp(d.name)[0] in args.scenes]

    if args.exp:
        exps = [d for d in exps if d.name == args.exp]
        if not exps:
            sys.exit(f"kein Experimentverzeichnis mit dem Namen {args.exp}")

    done = done_set(args.retry_failed)
    todo = [d for d in exps if d.name not in done]
    if args.limit is not None:
        todo = todo[:args.limit]

    print(f"{len(exps)} experiments, {len(done)} already done, {len(todo)} to run")
    if args.dry_run:
        for d in todo:
            print("  TODO", d.name)
        return 0
    if not todo:
        print("nothing to do")
        return 0

    ver = versions()
    print(f"versions: {ver}")

    for i, d in enumerate(todo, 1):
        if not cuda_ok():
            print("\nCUDA not available, stopping. Fix the GPU and re-run to resume.")
            return 1
        print(f"\n[{i}/{len(todo)}] {d.name} ...", flush=True)
        row = run_one(d, ver)
        append_row(row)
        tag = row["status"]
        extra = (f"PSNR {row.get('psnr','?')}  faces {row.get('n_faces','?')}  "
                 f"{row.get('total_s','?')}s") if tag == "ok" else ""
        print(f"    {tag}  {extra}", flush=True)

    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
