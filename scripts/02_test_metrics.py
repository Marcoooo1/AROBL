#!/usr/bin/env python
"""Validiert das Metrikmodul an kuenstlich verfaelschten Kopien der Referenz.

Der Punkt: bei jedem Fall ist die richtige Antwort vorher bekannt. Wenn die
Metriken hier nicht stimmen, liegt es am Code und nicht am NeRF.

Aufruf:
    python scripts/02_test_metrics.py data/reference/testobjekt_1.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nerfpipe.metrics import geometry as G  # noqa: E402

SAMPLES = 30_000


def rt(angles_deg, translation) -> np.ndarray:
    """Starre Transformation aus Eulerwinkeln und Verschiebung."""
    rx, ry, rz = np.radians(angles_deg)
    t = (trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz"))
    t[:3, 3] = translation
    return t


def banner(name: str) -> None:
    print(f"\n{name}\n" + "-" * 72)


def row(label: str, value: float, expect: str, ok: bool) -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"  {mark}  {label:<26} {value:>10.4f} mm   erwartet {expect}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("--samples", type=int, default=SAMPLES)
    args = ap.parse_args()

    ref = G.load_mesh(args.reference)
    diag = float(np.linalg.norm(ref.extents))
    print(f"Referenz: {args.reference.name}, Bounding Box "
          f"{' x '.join(f'{e:.1f}' for e in ref.extents)}, Diagonale {diag:.1f} mm")

    passed: list[bool] = []
    kw = dict(samples=args.samples, seed=1)

    # --- Fall 1: identisch -------------------------------------------------
    banner("Fall 1: identische Kopie, keine Verfaelschung")
    m = G.compare(ref.copy(), ref, **kw)
    passed.append(row("Chamfer Mittel", m.chamfer_mean, "< 0.05", m.chamfer_mean < 0.05))
    passed.append(row("Hausdorff", m.hausdorff, "< 0.5", m.hausdorff < 0.5))

    # --- Fall 2: verschoben und verdreht -----------------------------------
    banner("Fall 2: um 17/5/31 Grad verdreht und um (12, -8, 25) verschoben")
    moved = ref.copy()
    moved.apply_transform(rt((17, 5, 31), (12, -8, 25)))
    m = G.compare(moved, ref, **kw)
    passed.append(row("Chamfer Mittel", m.chamfer_mean, "< 0.05", m.chamfer_mean < 0.05))
    passed.append(row("Hausdorff", m.hausdorff, "< 0.5", m.hausdorff < 0.5))

    # --- Fall 3: gleichmaessiges Rauschen ----------------------------------
    for sigma in (0.1, 0.5):
        banner(f"Fall 3: Punktrauschen mit Standardabweichung {sigma} mm")
        noisy = ref.copy()
        rng = np.random.default_rng(7)
        noisy.vertices = noisy.vertices + rng.normal(0, sigma, noisy.vertices.shape)
        m = G.compare(noisy, ref, **kw)
        lo, hi = 0.3 * sigma, 2.5 * sigma
        passed.append(row("Chamfer Mittel", m.chamfer_mean,
                          f"{lo:.2f} bis {hi:.2f}", lo < m.chamfer_mean < hi))

    # --- Fall 4: Massstabsfehler -------------------------------------------
    banner("Fall 4: um Faktor 1.05 zu gross, also 5 Prozent Massstabsfehler")
    scaled = ref.copy()
    scaled.apply_scale(1.05)
    scaled.apply_transform(rt((10, 20, 30), (5, 5, 5)))

    m_rigid = G.compare(scaled, ref, allow_scale=False, **kw)
    m_sim = G.compare(scaled, ref, allow_scale=True, **kw)

    expect_rigid = 0.05 * diag / 4
    ok_rigid = 0.3 * expect_rigid < m_rigid.chamfer_mean < 3 * expect_rigid
    passed.append(row("starr, Chamfer", m_rigid.chamfer_mean,
                      f"um {expect_rigid:.2f}", ok_rigid))
    passed.append(row("Aehnlichkeit, Chamfer", m_sim.chamfer_mean,
                      "< 0.05", m_sim.chamfer_mean < 0.05))

    recovered = m_sim.scale_factor
    ok_scale = abs(recovered - 1.05) < 0.005
    print(f"  {'OK  ' if ok_scale else 'FAIL'}  "
          f"{'Massstab zurueckgewonnen':<26} {recovered:>10.4f}      erwartet 1.0500")
    passed.append(ok_scale)

    # --- Fall 5: fehlendes Stueck ------------------------------------------
    banner("Fall 5: obere Haelfte abgeschnitten, unvollstaendige Rekonstruktion")
    cut = ref.copy()
    z_mid = cut.bounds[:, 2].mean()
    # Flaechen oberhalb der Mitte entfernen. Bewusst ueber die Flaechen
    # schwerpunkte statt slice_plane, das wuerde shapely voraussetzen.
    keep = cut.triangles_center[:, 2] < z_mid
    cut.update_faces(keep)
    cut.remove_unreferenced_vertices()
    m = G.compare(cut, ref, **kw)
    ok_asym = m.completeness > 3 * m.accuracy
    print(f"  {'OK  ' if ok_asym else 'FAIL'}  "
          f"{'completeness > accuracy':<26} "
          f"{m.completeness:>10.4f} gegen {m.accuracy:.4f}")
    print("        completeness misst fehlende Teile, accuracy misst "
          "falsch platzierte.")
    passed.append(ok_asym)

    print("\n" + "=" * 72)
    print(f"{sum(passed)} von {len(passed)} Pruefungen bestanden")
    print("=" * 72 + "\n")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
