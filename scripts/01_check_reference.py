#!/usr/bin/env python
"""Prueft eine Referenz STL auf Eignung als Ground Truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
_status: list[str] = []


def say(level: str, text: str, hint: str = "") -> None:
    _status.append(level)
    print(f"  {level}  {text}")
    if hint:
        print(f"        -> {hint}")


def check(path: Path, nominal: list[float] | None) -> bool:
    print(f"\n{path.name}")
    print("-" * 68)
    _status.clear()

    try:
        # process=True fuehrt doppelte Punkte zusammen. STL speichert jedes
        # Dreieck getrennt, ohne das waere jedes Netz scheinbar defekt.
        mesh = trimesh.load_mesh(path, process=True)
        mesh.merge_vertices()
    except Exception as exc:
        say(FAIL, f"nicht ladbar: {exc}")
        return False

    if isinstance(mesh, trimesh.Scene):
        say(FAIL, "Datei enthaelt eine Szene, nicht ein einzelnes Mesh",
            "In SolidWorks jede Komponente als eigene STL exportieren.")
        return False

    say(OK, f"{len(mesh.faces)} Dreiecke, {len(mesh.vertices)} Punkte")

    if mesh.is_watertight:
        say(OK, "wasserdicht")
    else:
        say(FAIL, "nicht wasserdicht",
            "Loecher im Netz. Volumen und Chamfer werden unzuverlaessig.")

    if mesh.is_winding_consistent:
        say(OK, "Normalen konsistent")
    else:
        say(FAIL, "Normalen inkonsistent", "Innen und aussen nicht eindeutig.")

    n_components = len(mesh.split(only_watertight=False))
    if n_components == 1:
        say(OK, "eine zusammenhaengende Komponente")
    else:
        say(FAIL, f"{n_components} Komponenten",
            "Lose Teile entfernen, sonst verfaelschen sie die Chamfer Distanz.")

    edges = mesh.edges_sorted.reshape((-1, 2))
    _, counts = np.unique(edges, axis=0, return_counts=True)
    bad_edges = int((counts != 2).sum())
    if bad_edges == 0:
        say(OK, "keine Non Manifold Kanten")
    else:
        say(FAIL, f"{bad_edges} Non Manifold Kanten", "Netz ist topologisch defekt.")

    degenerate = int((~mesh.nondegenerate_faces()).sum())
    if degenerate == 0:
        say(OK, "keine entarteten Dreiecke")
    else:
        say(WARN, f"{degenerate} entartete Dreiecke",
            "Flaeche nahe null. Stoert manche Metrikbibliotheken.")

    extents = mesh.extents
    say(OK, "Bounding Box " + " x ".join(f"{e:.2f}" for e in sorted(extents, reverse=True)))

    if mesh.is_watertight:
        vol = mesh.volume
        if vol > 0:
            say(OK, f"Volumen {vol/1000:.2f} cm3, Oberflaeche {mesh.area/100:.2f} cm2")
        else:
            say(FAIL, f"Volumen negativ ({vol:.1f})", "Normalen zeigen nach innen.")

    if max(extents) < 5:
        say(WARN, "groesste Ausdehnung unter 5",
            "Datei ist vermutlich in Metern oder Zoll, nicht in Millimeter.")
    elif max(extents) > 1000:
        say(WARN, "groesste Ausdehnung ueber 1000", "Einheit pruefen.")

    if nominal:
        for g, w in zip(sorted(extents, reverse=True), sorted(nominal, reverse=True)):
            d = g - w
            level = OK if abs(d) < 0.05 else FAIL
            say(level, f"Nennmass {w:.2f} gegen {g:.2f}, Abweichung {d:+.3f}")

    # Lange Kanten auf einer ebenen Flaeche sind richtig. Relevant ist nur
    # die Tesselierung von Rundungen, also Kanten mit kleinem Knickwinkel.
    angles = mesh.face_adjacency_angles
    curved = (angles > np.radians(0.5)) & (angles < np.radians(40))

    if not curved.any():
        say(OK, "keine gekruemmten Flaechen erkannt, Aufloesung unkritisch")
    else:
        theta = angles[curved]
        # Facettenbreite quer zur geteilten Kante: Abstand der Flaechen
        # schwerpunkte, projiziert senkrecht zur Kantenrichtung. Faktor 1.5
        # korrigiert die Schwerpunktlage, gegen exakte Zylinderwerte geprueft.
        pair = mesh.face_adjacency[curved]
        delta = mesh.triangles_center[pair[:, 1]] - mesh.triangles_center[pair[:, 0]]
        ev = mesh.face_adjacency_edges[curved]
        edge_dir = mesh.vertices[ev[:, 1]] - mesh.vertices[ev[:, 0]]
        edge_dir = edge_dir / np.linalg.norm(edge_dir, axis=1, keepdims=True)
        width = 1.5 * np.linalg.norm(
            delta - (delta * edge_dir).sum(1, keepdims=True) * edge_dir, axis=1
        )
        worst = float((width * theta / 8.0).max())
        say(OK, f"{curved.sum()} Kanten auf Rundungen, "
                f"groesster Knick {np.degrees(theta.max()):.1f} Grad")
        level = OK if worst < 0.02 else (WARN if worst < 0.05 else FAIL)
        say(level, f"geschaetzte maximale Sehnenabweichung {worst:.4f} mm",
            "" if level == OK else
            "Im STL Export Abweichungstoleranz auf 0.01 mm und "
            "Winkeltoleranz auf 1 Grad setzen.")

    return FAIL not in _status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--nominal", nargs=3, type=float, default=None,
                    metavar=("A", "B", "C"),
                    help="erwartete Bounding Box in mm, Reihenfolge egal")
    args = ap.parse_args()

    results = [check(f, args.nominal) for f in args.files]
    print()
    for f, ok in zip(args.files, results):
        print(f"{'bestanden' if ok else 'DURCHGEFALLEN':>14}  {f.name}")
    print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
