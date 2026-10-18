#!/usr/bin/env python
"""Diagnose der Transformkette und der Mesh Ausdehnung.

Trennt zwei Fehlerquellen, die sich in einer nackten Bounding Box nicht
unterscheiden lassen:
  a) die Transformation nach DTU mm ist falsch
  b) die Transformation stimmt, das Mesh enthaelt aber weit entfernte
     Floater, die jede Bounding Box und jeden Mittelwert dominieren

Dazu werden die Eckpunkte in den normierten Rahmen zurueckgerechnet, in dem
das Objekt per Konstruktion in der Einheitskugel sitzt, und die Verteilung
der Radien ausgegeben. Liegt der Hauptteil unter 1, stimmt die Kette und
das Problem sind die Floater.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_geometry import (MESH_DIR, PREPARED_ROOT, parse_exp,
                           dataparser_to_ns, latest_dataparser, find_scale_mat)


def diag(exp: str, prepared_root: Path = PREPARED_ROOT) -> None:
    scene = parse_exp(exp)[0]
    ply = MESH_DIR / exp / "poisson_mesh.ply"
    print(f"\n{exp}")
    print(f"  Datei {ply}  {ply.stat().st_size/1e6:.1f} MB")

    dp = latest_dataparser(exp)
    A = dataparser_to_ns(dp)
    S = find_scale_mat(scene, prepared_root)
    print(f"  dataparser {dp}")
    print(f"    scale*R Diagonale {np.round(np.diag(A)[:3], 6)}")
    print(f"    Translation       {np.round(A[:3, 3], 6)}")
    print(f"    ist Identitaet    {np.allclose(A, np.eye(4), atol=1e-9)}")
    print(f"  scale_mat Radius {S[0,0]:.3f} mm  Zentrum {np.round(S[:3,3], 2)}")

    mesh = trimesh.load_mesh(ply, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    v = np.asarray(mesh.vertices, dtype=np.float64)
    print(f"  Eckpunkte {len(v):,}  Flaechen {len(mesh.faces):,}")

    # zurueck in den normierten Rahmen
    v_norm = trimesh.transform_points(v, np.linalg.inv(A))
    r = np.linalg.norm(v_norm, axis=1)
    qs = [50, 90, 95, 99, 99.9, 100]
    print("  Radius im normierten Rahmen (Objekt sollte unter 1 liegen):")
    print("    " + "  ".join(f"p{q:g} {np.percentile(r, q):.3f}" for q in qs))
    for thr in (1.0, 1.5, 2.0):
        print(f"    Anteil mit Radius <= {thr}: {(r <= thr).mean()*100:.2f} %")

    # robuste Ausdehnung: nur der Kern innerhalb der Einheitskugel
    core = v_norm[r <= 1.0]
    if len(core) == 0:
        print("    KEIN Eckpunkt in der Einheitskugel, Transformation verdaechtig")
        return
    ext_norm = core.max(0) - core.min(0)
    ext_mm = ext_norm * S[0, 0]
    print(f"  Kern in der Einheitskugel: {len(core):,} Eckpunkte")
    print(f"    Ausdehnung normiert {np.round(ext_norm, 3)}")
    print(f"    Ausdehnung in DTU mm {np.round(ext_mm, 1)}   <- sollte ~100 bis 400 mm sein")


def main() -> int:
    exps = sys.argv[1:] or sorted(d.name for d in MESH_DIR.iterdir()
                                  if d.is_dir() and parse_exp(d.name))
    for e in exps:
        diag(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
