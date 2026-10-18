#!/usr/bin/env python
"""Bestimmt, in welchem Koordinatenrahmen das exportierte Mesh liegt.

Drei Moeglichkeiten, die sich an den Radien unterscheiden lassen:
  A  normierter Rahmen, Objekt in der Einheitskugel   -> |v| um 1
  B  bereits DTU Millimeter                           -> |v - Zentrum| < Radius
  C  normierter Rahmen, aber nur Hintergrundmuell     -> |v| >> Kameraradius

Als Referenz dienen die Kamerapositionen aus der transforms.json des Laufs.
Die liegen per Konstruktion im selben Rahmen wie die Trainingsdaten, also im
Rahmen, in dem Nerfstudio auch exportiert.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_geometry import MESH_DIR, PREPARED_ROOT, parse_exp, find_scale_mat

EXP_ROOT = Path("data/dtu/experiments")


def pct(label, arr, scale=1.0):
    q = [0, 50, 90, 100]
    print(f"    {label:<34} " + "  ".join(
        f"p{v:g} {np.percentile(arr, v)*scale:.2f}" for v in q))


def diag(exp: str) -> None:
    scene = parse_exp(exp)[0]
    print(f"\n{exp}")

    S = find_scale_mat(scene, PREPARED_ROOT)
    centre, radius = S[:3, 3], S[0, 0]
    print(f"  scale_mat: Radius {radius:.2f} mm, Zentrum {np.round(centre,2)}")
    print(f"  Objektpunkte in DTU mm haetten |v| zwischen "
          f"{np.linalg.norm(centre)-radius:.0f} und {np.linalg.norm(centre)+radius:.0f}")

    tj = EXP_ROOT / exp / "transforms.json"
    cams = np.array([np.array(f["transform_matrix"])[:3, 3]
                     for f in json.loads(tj.read_text())["frames"]])
    cam_r = np.linalg.norm(cams, axis=1)
    print(f"  Kameras aus {tj.name}: {len(cams)} Stueck")
    pct("|Kameraposition|", cam_r)
    pct("|Kamera - scale_mat Zentrum|", np.linalg.norm(cams - centre, axis=1))

    mesh = trimesh.load_mesh(MESH_DIR / exp / "poisson_mesh.ply", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    v = np.asarray(mesh.vertices, dtype=np.float64)
    vr = np.linalg.norm(v, axis=1)
    vcr = np.linalg.norm(v - centre, axis=1)
    print(f"  Mesh: {len(v):,} Eckpunkte")
    pct("|Eckpunkt|", vr)
    pct("|Eckpunkt - scale_mat Zentrum|", vcr)

    cam_med, v_med, vc_med = np.median(cam_r), np.median(vr), np.median(vcr)
    print("  Befund:")
    if v_med < 2.0:
        print("    A  normierter Rahmen, Objekt in der Einheitskugel. Kette stimmt.")
    elif vc_med < 1.5 * radius and v_med > 5 * cam_med:
        print("    B  Mesh liegt BEREITS in DTU mm. scale_mat darf nicht erneut")
        print("       angewendet werden, die Transformation ist dann die Identitaet.")
    elif v_med > 5 * cam_med:
        print("    C  Mesh liegt weit jenseits der Kameras, also Hintergrundmuell")
        print("       im normierten Rahmen. Export muss begrenzt werden.")
    else:
        print("    unklar, Zahlen von Hand ansehen")
    print(f"    Verhaeltnis Median |Eckpunkt| zu Median |Kamera|: {v_med/cam_med:.1f}")


def main() -> int:
    exps = sys.argv[1:] or sorted(d.name for d in MESH_DIR.iterdir()
                                  if d.is_dir() and parse_exp(d.name))
    for e in exps:
        diag(e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
