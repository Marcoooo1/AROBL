#!/usr/bin/env python
"""Geometrie Auswertung eines Laufs oder aller Laeufe (Teil A).

Fuer jeden Lauf liegt das von Nerfstudio exportierte Mesh im internen
Szenenrahmen von Nerfstudio vor. Dieses Skript bringt es zurueck in die
DTU Weltkoordinaten (Millimeter), damit es mit dem Laserscan vergleichbar
ist, und wertet es nach dem DTU Protokoll aus (nerfpipe.metrics.dtu_eval).

Transformkette, das ist der entscheidende Schritt:
    Mesh im Nerfstudio Rahmen
      -> inverse dataparser Transformation (Nerfstudio schreibt sie je Lauf)
         ergibt den normierten Eingaberahmen (Einheitskugel)
      -> scale_mat der Szene
         ergibt DTU Weltkoordinaten in Millimeter
Beide Transformationen werden zu einer 4x4 Matrix zusammengefasst und auf
das Mesh angewendet. Geprueft wird das ueber M @ A == scale_mat.

Ergebnisse gehen in results/geometry.csv, eine Zeile je Lauf, Schluessel exp.
Die Rendering Metriken bleiben in results/metrics.csv. Der Korrelationsschritt
fuehrt beide ueber die Spalte exp zusammen.

Die DTU Grundwahrheit muss unter --dtu-eval-dir liegen. Fehlt sie, schreibt
der Lauf eine Zeile mit status no_gt und das Skript nennt, was zu holen ist.

Aufruf:
    python scripts/eval_geometry.py --exp scan24_n05_s0
    python scripts/eval_geometry.py --all
    python scripts/eval_geometry.py --all --save-transformed
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nerfpipe.metrics import dtu_eval as D  # noqa: E402

MESH_DIR = Path("results/meshes")
OUT_ROOT = Path("outputs/exp")
PREPARED_ROOT = Path("data/dtu/prepared")
DTU_EVAL_DIR = Path("data/dtu/eval")
GEOM_CSV = Path("results/geometry.csv")

FIELDS = [
    "exp", "scene", "n", "seed", "status",
    "accuracy", "completeness", "chamfer", "f_score",
    "precision", "recall", "fscore_thresh",
    "n_data", "n_gt",
]


def parse_exp(name: str):
    m = re.match(r"(scan\d+)_n(\d+)_s(\d+)$", name)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def scan_number(scene: str) -> int:
    return int(re.search(r"\d+", scene).group())


# --------------------------------------------------------------------------
# Transformkette
# --------------------------------------------------------------------------

def dataparser_to_ns(dp_path: Path) -> np.ndarray:
    """4x4 Matrix A, die den Eingaberahmen in den Nerfstudio Rahmen abbildet.

    Nerfstudio: p_ns = scale * (transform @ [p_in, 1]).
    Also A = [[scale*R | scale*t], [0 0 0 1]].
    """
    d = json.loads(dp_path.read_text())
    T = np.array(d["transform"], dtype=np.float64)   # 3x4
    s = float(d["scale"])
    A = np.eye(4)
    A[:3, :3] = s * T[:3, :3]
    A[:3, 3] = s * T[:3, 3]
    return A


def compose_to_dtu(dp_path: Path, scale_mat: np.ndarray) -> np.ndarray:
    """4x4 Matrix M, die den Nerfstudio Rahmen nach DTU mm abbildet.

    M = scale_mat @ inverse(A). Dann ist M @ A == scale_mat, der Weg vom
    Eingaberahmen direkt nach DTU, das ist der Round Trip Test.
    """
    A = dataparser_to_ns(dp_path)
    return scale_mat @ np.linalg.inv(A)


def latest_dataparser(exp: str) -> Path:
    runs = sorted((OUT_ROOT / exp / "nerfacto").glob("*/dataparser_transforms.json"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"keine dataparser_transforms.json fuer {exp}")
    return runs[-1]


def find_scale_mat(scene: str, prepared_root: Path) -> np.ndarray:
    """scale_mat.npy der Szene finden, egal ob der Ordner scan24 oder
    dtu_scan24 heisst. Zuordnung ueber die Scan Nummer im Ordnernamen."""
    num = scan_number(scene)
    for p in sorted(prepared_root.glob("*/scale_mat.npy")):
        if scan_number(p.parent.name) == num:
            return np.load(p)
    raise FileNotFoundError(
        f"keine scale_mat.npy fuer {scene} unter {prepared_root}. "
        f"Erst scripts/dtu_prepare.py laufen lassen."
    )


# --------------------------------------------------------------------------
# ein Lauf
# --------------------------------------------------------------------------

def mesh_to_dtu(exp: str, scene: str, prepared_root: Path,
                crop_radius: float | None) -> trimesh.Trimesh:
    ply = MESH_DIR / exp / "poisson_mesh.ply"
    if not ply.exists():
        raise FileNotFoundError(f"kein Mesh: {ply}")
    mesh = trimesh.load_mesh(ply, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    scale_mat = find_scale_mat(scene, prepared_root)
    dp = latest_dataparser(exp)
    M = compose_to_dtu(dp, scale_mat)

    # Croppen passiert im normierten Rahmen, vor der DTU Transformation.
    # Das Objekt sitzt dort in der Einheitskugel, weit entfernte Floater
    # liegen ausserhalb und koennen hier weg. Standard aus, die ObsMask
    # erledigt Ausreisser ohnehin.
    if crop_radius is not None:
        A = dataparser_to_ns(dp)
        v_norm = trimesh.transform_points(mesh.vertices, np.linalg.inv(A))
        inside = np.linalg.norm(v_norm, axis=1) <= crop_radius
        keep = inside[mesh.faces].all(axis=1)
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()

    mesh.apply_transform(M)
    return mesh


def eval_one(exp: str, gt, prepared_root: Path,
             crop_radius: float | None, save_transformed: bool,
             density: float, fscore_thresh: float) -> dict:
    scene, n, seed = parse_exp(exp)
    row = {"exp": exp, "scene": scene, "n": n, "seed": seed}
    try:
        mesh = mesh_to_dtu(exp, scene, prepared_root, crop_radius)
    except Exception as e:
        row["status"] = "failed:transform"
        print(f"    {e}")
        return row

    if save_transformed:
        mesh.export(MESH_DIR / exp / "poisson_mesh_dtu_mm.ply")

    if gt is None:
        row["status"] = "no_gt"
        return row

    try:
        m = D.evaluate_mesh(mesh, gt, density=density, fscore_thresh=fscore_thresh)
    except Exception as e:
        row["status"] = "failed:eval"
        print(f"    {e}")
        return row

    row.update(m.as_dict())
    row["status"] = "ok"
    return row


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def done_set() -> set:
    if not GEOM_CSV.exists():
        return set()
    with open(GEOM_CSV) as f:
        return {r["exp"] for r in csv.DictReader(f) if r["status"] == "ok"}


def append_row(row: dict) -> None:
    GEOM_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not GEOM_CSV.exists()
    full = {k: row.get(k, "") for k in FIELDS}
    with open(GEOM_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(full)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp", help="ein Lauf, z.B. scan24_n05_s0")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--dtu-eval-dir", type=Path, default=DTU_EVAL_DIR)
    ap.add_argument("--prepared-root", type=Path, default=PREPARED_ROOT)
    ap.add_argument("--crop-radius", type=float, default=None,
                    help="Mesh im normierten Rahmen auf diesen Radius croppen")
    ap.add_argument("--save-transformed", action="store_true",
                    help="das nach DTU mm transformierte Mesh ablegen")
    ap.add_argument("--density", type=float, default=D.DOWNSAMPLE_DENSITY)
    ap.add_argument("--fscore-thresh", type=float, default=D.FSCORE_THRESH)
    ap.add_argument("--redo", action="store_true", help="auch fertige neu rechnen")
    args = ap.parse_args()

    if args.exp:
        exps = [args.exp]
    else:
        exps = sorted(d.name for d in MESH_DIR.iterdir()
                      if d.is_dir() and parse_exp(d.name))
    if args.scenes:
        exps = [e for e in exps if parse_exp(e)[0] in args.scenes]
    if not args.redo:
        done = done_set()
        exps = [e for e in exps if e not in done]

    if not exps:
        print("nichts zu tun")
        return 0

    # Grundwahrheit je Szene einmal laden und wiederverwenden.
    gt_cache = {}

    def gt_for(scene: str):
        num = scan_number(scene)
        if num not in gt_cache:
            try:
                gt_cache[num] = D.load_ground_truth(args.dtu_eval_dir, num)
            except FileNotFoundError as e:
                print(f"  {e}")
                gt_cache[num] = None
        return gt_cache[num]

    print(f"{len(exps)} Lauf/Laeufe auszuwerten")
    any_gt = False
    for i, exp in enumerate(exps, 1):
        scene = parse_exp(exp)[0]
        gt = gt_for(scene)
        any_gt = any_gt or gt is not None
        print(f"[{i}/{len(exps)}] {exp} ...", flush=True)
        row = eval_one(exp, gt, args.prepared_root, args.crop_radius,
                       args.save_transformed, args.density, args.fscore_thresh)
        append_row(row)
        if row["status"] == "ok":
            print(f"    chamfer {row['chamfer']:.4f} mm  f@{args.fscore_thresh:g} "
                  f"{row['f_score']:.4f}  acc {row['accuracy']:.4f}  "
                  f"comp {row['completeness']:.4f}", flush=True)
        else:
            print(f"    {row['status']}", flush=True)

    if not any_gt:
        print("\nKeine DTU Grundwahrheit gefunden, nur die Transformation lief.")
        print(f"Erwartet unter {args.dtu_eval_dir}:")
        print("  Points/stl/stl0NN_total.ply   ObsMask/ObsMaskNN_10.mat   ObsMask/PlaneNN.mat")
        print("Quelle: offizielle DTU MVS Seite, Archive SampleSet und Points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
