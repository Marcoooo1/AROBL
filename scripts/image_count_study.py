#!/usr/bin/env python3
"""DTU image count study: 3 scenes, 10 image counts, one seed."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs" / "image_count_study"
PREPARED = ROOT / "data" / "dtu" / "prepared"
GT = ROOT / "data" / "dtu" / "eval"

SCENES = ("scan24", "scan37", "scan55")
COUNTS = (49, 45, 40, 35, 30, 25, 20, 15, 10, 5)

STEPS = 12000
BATCH = 4096
POINTS = 1_000_000
SEED = 0

COLUMNS = (
    "exp", "scene", "n", "seed",
    "independent_eval", "test_overlap",
    "train_s", "total_s",
    "psnr", "ssim", "lpips",
    "n_vertices", "n_faces",
    "accuracy", "completeness", "chamfer",
    "f_score", "precision", "recall",
    "geometry_status", "status", "error",
    "updated_utc"
)


# --------------------------------------------------
# CSV output and summary
# --------------------------------------------------

def load_rows() -> dict[str, dict]:
    path = STUDY / "metrics.csv"

    if not path.exists():
        return {}

    with path.open(newline="") as f:
        return {
            row["exp"]: row
            for row in csv.DictReader(f)
        }


def save_rows(rows: dict[str, dict]) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)

    dest = STUDY / "metrics.csv"
    temp = STUDY / "metrics.csv.tmp"

    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=COLUMNS
        )
        writer.writeheader()

        for name in sorted(rows):
            writer.writerow({
                key: rows[name].get(key, "")
                for key in COLUMNS
            })

    temp.replace(dest)

    summary = STUDY / "summary_by_n.csv"

    with summary.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "n",
            "ok",
            "mean_chamfer_mm",
            "median_chamfer_mm",
            "mean_f_score",
            "mean_psnr_heldout"
        ])

        for n in COUNTS:
            good = [
                r for r in rows.values()
                if r.get("status") == "ok"
                and int(r["n"]) == n
            ]

            chamfer = [
                float(r["chamfer"]) for r in good
            ]
            fscores = [
                float(r["f_score"]) for r in good
            ]
            psnr = [
                float(r["psnr"]) for r in good
                if str(r.get("psnr", ""))
            ]

            writer.writerow([
                n,
                len(good),
                mean(chamfer) if chamfer else "",
                median(chamfer) if chamfer else "",
                mean(fscores) if fscores else "",
                mean(psnr) if psnr else ""
            ])


# --------------------------------------------------
# Nested image splits
# --------------------------------------------------

def link_to(target: Path, link: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(
            f"Existing path is not a symlink: {link}"
        )

    link.symlink_to(
        os.path.relpath(target, link.parent),
        target_is_directory=target.is_dir()
    )


def make_split(scene: str, n: int) -> tuple[Path, int]:
    src = PREPARED / scene

    base = json.loads(
        (src / "transforms.json").read_text()
    )

    paths = [
        frame["file_path"]
        for frame in base["frames"]
    ]

    if len(paths) != 49 or len(set(paths)) != 49:
        raise ValueError(
            f"{scene}: Expected 49 distinct images, "
            f"found {len(paths)}"
        )

    # Six fixed evaluation views per scene.
    test_idx = sorted({
        round(i * 48 / 5)
        for i in range(6)
    })

    # First 43 images exclude the test views.
    pool = [
        i for i in range(49)
        if i not in test_idx
    ]

    random.Random(
        1000 + int(scene[4:])
    ).shuffle(pool)

    # Nested subsets.
    # N=45 and N=49 include some evaluation views.
    order = pool + test_idx

    train = [
        paths[i] for i in order[:n]
    ]

    test = [
        paths[i] for i in test_idx
    ]

    overlap = len(
        set(train).intersection(test)
    )

    exp = f"{scene}_n{n:02d}_s{SEED}"
    dst = STUDY / "splits" / exp

    dst.mkdir(parents=True, exist_ok=True)

    doc = dict(base)
    doc.update(
        train_filenames=train,
        val_filenames=test,
        test_filenames=test
    )

    (dst / "transforms.json").write_text(
        json.dumps(doc, indent=2)
    )

    for name in (
        "images",
        "masks",
        "scale_mat.npy"
    ):
        target = src / name

        if target.exists():
            link_to(target, dst / name)

        elif name in ("images", "scale_mat.npy"):
            raise FileNotFoundError(target)

    return dst, overlap


# --------------------------------------------------
# Commands and checkpoint handling
# --------------------------------------------------

def run_cmd(
    cmd: list[str],
    logfile: Path
) -> None:

    logfile.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with logfile.open("a") as log:
        log.write(
            "\n$ " + shlex.join(cmd) + "\n"
        )
        log.flush()

        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT
        )

        if result.returncode:
            raise RuntimeError(
                f"Command failed ({result.returncode}): "
                f"{shlex.join(cmd[:2])}"
            )


def complete_config(exp: str) -> Path | None:
    directory = (
        STUDY / "training" / exp / "nerfacto"
    )

    configs = sorted(
        directory.glob("*/config.yml"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    for cfg in configs:
        checkpoint = (
            cfg.parent
            / "nerfstudio_models"
            / f"step-{STEPS - 1:09d}.ckpt"
        )

        if checkpoint.is_file():
            return cfg

    return None


def ply_counts(ply: Path) -> tuple[int, int]:
    vertices = None
    faces = None

    with ply.open("rb") as f:
        for raw in f:
            if raw.startswith(b"element vertex "):
                vertices = int(raw.split()[-1])

            elif raw.startswith(b"element face "):
                faces = int(raw.split()[-1])

            elif raw.strip() == b"end_header":
                break

    if (
        vertices is None
        or faces is None
        or vertices == 0
        or faces == 0
    ):
        raise ValueError(
            f"Mesh is empty or invalid: {ply}"
        )

    return vertices, faces


# --------------------------------------------------
# DTU geometry evaluation
# --------------------------------------------------

GEOMETRY_CODE = """
import sys
from pathlib import Path
import scripts.eval_geometry as e

base = Path(sys.argv[1])

e.MESH_DIR = base / 'meshes'
e.OUT_ROOT = base / 'training'
e.GEOM_CSV = base / 'geometry.csv'

sys.argv = [
    'eval_geometry.py',
    '--exp', sys.argv[2],
    '--crop-radius', '1.0'
]

raise SystemExit(e.main())
"""


def geometry_row(exp: str) -> dict | None:
    path = STUDY / "geometry.csv"

    if not path.exists():
        return None

    with path.open(newline="") as f:
        matched = [
            r for r in csv.DictReader(f)
            if r["exp"] == exp
        ]

    return matched[-1] if matched else None


# --------------------------------------------------
# One complete experiment
# --------------------------------------------------

def run_one(
    scene: str,
    n: int,
    rows: dict[str, dict]
) -> None:

    exp = f"{scene}_n{n:02d}_s{SEED}"

    split, overlap = make_split(scene, n)

    row = rows.get(
        exp,
        {
            "exp": exp,
            "scene": scene,
            "n": n,
            "seed": SEED
        }
    )

    row["independent_eval"] = overlap == 0
    row["test_overlap"] = overlap

    log = STUDY / "logs" / f"{exp}.log"

    mesh = (
        STUDY / "meshes"
        / exp / "poisson_mesh.ply"
    )

    stage = "train"
    t0 = time.perf_counter()

    try:
        cfg = complete_config(exp)

        # ------------------------------------------
        # Train Nerfacto
        # ------------------------------------------

        if cfg is None:
            cmd = [
                "ns-train", "nerfacto",

                "--max-num-iterations", str(STEPS),
                "--machine.seed", str(SEED),
                "--vis", "tensorboard",

                "--steps-per-eval-image", "3000",
                "--steps-per-eval-batch", "3000",
                "--steps-per-save", "6000",

                "--pipeline.datamanager.images-on-gpu",
                "True",

                "--pipeline.datamanager.train-num-rays-per-batch",
                str(BATCH),

                "--output-dir",
                str(STUDY / "training"),

                "--experiment-name", exp,

                "--pipeline.model.predict-normals",
                "True",

                "--pipeline.model.background-color",
                "black",

                "--pipeline.model.camera-optimizer.mode",
                "off",

                "--pipeline.model.use-appearance-embedding",
                "False",

                "--pipeline.datamanager.pixel-sampler.ignore-mask",
                "True",

                "--data", str(split),

                "nerfstudio-data",

                "--orientation-method", "none",
                "--center-method", "none",
                "--auto-scale-poses", "False",
                "--mask-color", "0", "0", "0"
            ]

            started = time.perf_counter()

            run_cmd(cmd, log)

            row["train_s"] = round(
                time.perf_counter() - started
            )

            cfg = complete_config(exp)

            if cfg is None:
                raise RuntimeError(
                    "Final checkpoint missing"
                )

        # ------------------------------------------
        # Rendering evaluation
        # ------------------------------------------

        stage = "render_eval"

        if overlap == 0:
            eval_json = (
                STUDY / "evaluations"
                / f"{exp}.json"
            )

            if not eval_json.exists():
                eval_json.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                run_cmd([
                    "ns-eval",
                    "--load-config", str(cfg),
                    "--output-path", str(eval_json)
                ], log)

            metrics = json.loads(
                eval_json.read_text()
            )["results"]

            for key in ("psnr", "ssim", "lpips"):
                row[key] = float(metrics[key])

        else:
            # N=45 and N=49 have test overlap.
            # Do not report these as held-out metrics.
            for key in ("psnr", "ssim", "lpips"):
                row[key] = ""

        # ------------------------------------------
        # Poisson mesh export
        # ------------------------------------------

        stage = "mesh_export"

        if (
            not mesh.is_file()
            or mesh.stat().st_size == 0
        ):
            mesh.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            run_cmd([
                "ns-export", "poisson",
                "--load-config", str(cfg),

                "--output-dir", str(mesh.parent),
                "--texture-method", "point_cloud",

                "--num-points", str(POINTS),

                "--obb-center", "0", "0", "0",
                "--obb-rotation", "0", "0", "0",
                "--obb-scale", "2", "2", "2"
            ], log)

        row["n_vertices"], row["n_faces"] = (
            ply_counts(mesh)
        )

        # ------------------------------------------
        # DTU geometry evaluation
        # ------------------------------------------

        stage = "geometry"

        geo = geometry_row(exp)

        if geo is None or geo["status"] != "ok":
            run_cmd([
                sys.executable,
                "-c",
                GEOMETRY_CODE,
                str(STUDY),
                exp
            ], log)

            geo = geometry_row(exp)

        if geo is None or geo["status"] != "ok":
            raise RuntimeError(
                f"Geometry evaluation failed: {geo}"
            )

        row["geometry_status"] = "ok"

        for key in (
            "accuracy",
            "completeness",
            "chamfer",
            "f_score",
            "precision",
            "recall"
        ):
            row[key] = geo[key]

        row["status"] = "ok"
        row["error"] = ""

        print(
            f"  OK | Chamfer "
            f"{float(row['chamfer']):.3f} mm "
            f"| Faces {row['n_faces']}",
            flush=True
        )

    except KeyboardInterrupt:
        row["status"] = f"interrupted:{stage}"
        row["error"] = "Interrupted by user"
        raise

    except Exception as exc:
        row["status"] = f"failed:{stage}"
        row["error"] = str(exc)

        with log.open("a") as f:
            f.write(f"\nERROR: {exc}\n")

        print(
            f"  FAILED at {stage}: {exc}",
            flush=True
        )

    finally:
        row["total_s"] = round(
            time.perf_counter() - t0
        )

        row["updated_utc"] = (
            datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
        )

        rows[exp] = row
        save_rows(rows)


# --------------------------------------------------
# Environment checks
# --------------------------------------------------

def check_setup(scenes: list[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    from nerfstudio.exporter.exporter_utils import (
        generate_point_cloud
    )

    # Verify the deduplication patch.
    if "keep_indices" not in inspect.getsource(
        generate_point_cloud
    ):
        raise RuntimeError(
            "Nerfstudio duplicate point patch missing"
        )

    for scene in scenes:
        num = int(scene[4:])

        needed = (
            PREPARED / scene / "transforms.json",

            GT / "Points" / "stl"
            / f"stl{num:03d}_total.ply",

            GT / "ObsMask"
            / f"ObsMask{num}_10.mat",

            GT / "ObsMask"
            / f"Plane{num}.mat"
        )

        for path in needed:
            if not path.is_file():
                raise FileNotFoundError(path)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=SCENES,
        default=list(SCENES)
    )

    parser.add_argument(
        "--exp",
        help="Run only one experiment"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    cases = [
        (scene, n)
        for scene in args.scenes
        for n in COUNTS
    ]

    if args.exp:
        cases = [
            (scene, n)
            for scene, n in cases
            if f"{scene}_n{n:02d}_s{SEED}" == args.exp
        ]

        if not cases:
            parser.error(
                f"Experiment not found: {args.exp}"
            )

    rows = load_rows()

    cases = [
        (scene, n)
        for scene, n in cases
        if rows.get(
            f"{scene}_n{n:02d}_s{SEED}", {}
        ).get("status") != "ok"
    ]

    if args.limit is not None:
        cases = cases[:args.limit]

    print(f"{len(cases)} experiments remaining")

    for scene, n in cases:
        print(
            f"  {scene}_n{n:02d}_s{SEED}: "
            f"{n} training images"
        )

    if args.dry_run or not cases:
        return

    check_setup(args.scenes)

    for i, (scene, n) in enumerate(cases, 1):
        print(
            f"[{i}/{len(cases)}] "
            f"{scene}_n{n:02d}_s{SEED}",
            flush=True
        )

        run_one(scene, n, rows)

    print(
        "Results:",
        STUDY / "metrics.csv"
    )


if __name__ == "__main__":
    main()