"""Geometrische Metriken zwischen rekonstruiertem Mesh und Referenz.

Trennung nach Projektvorgabe: dieses Modul kennt nur Geometrie. Rendering
Metriken (PSNR, SSIM, LPIPS) gehoeren bewusst nicht hierher.

Ablauf:
    1. Beide Meshes gleichmaessig auf der Oberflaeche abtasten.
    2. Registrierung per ICP auf die Referenz.
       rigid      starre Transformation, Massstab bleibt erhalten
       similarity zusaetzlich ein globaler Massstabsfaktor
    3. Distanzmetriken auf den registrierten Punktwolken.

Zur Massstabsfrage: 'rigid' misst Form UND Massabweichung, setzt aber
voraus dass das Mesh bereits metrisch skaliert ist. 'similarity' entfernt
den Massstabsfehler und misst nur noch die Form. Beide Werte berichten,
die Differenz ist selbst ein Ergebnis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

DEFAULT_SAMPLES = 100_000


# --------------------------------------------------------------------------
# Laden und Abtasten
# --------------------------------------------------------------------------

def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Laedt ein Mesh und fuehrt doppelte Punkte zusammen.

    STL speichert jedes Dreieck mit eigenen Eckpunkten. Ohne Zusammenfuehren
    ist jede Topologiepruefung wertlos.
    """
    mesh = trimesh.load_mesh(Path(path), process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.merge_vertices()
    return mesh


def sample_surface(mesh: trimesh.Trimesh, n: int = DEFAULT_SAMPLES,
                   seed: int = 0) -> np.ndarray:
    """Gleichmaessig flaechengewichtete Punkte auf der Oberflaeche."""
    rng = np.random.default_rng(seed)
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(2**31)))
    return np.asarray(points, dtype=np.float64)


# --------------------------------------------------------------------------
# Registrierung
# --------------------------------------------------------------------------

def _principal_frame(points: np.ndarray) -> np.ndarray:
    """Transformation, die Schwerpunkt in den Ursprung und Hauptachsen auf
    die Koordinatenachsen legt."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    rot = vh
    if np.linalg.det(rot) < 0:          # Spiegelung vermeiden
        rot[2] *= -1
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = -rot @ centroid
    return transform


def register(source: np.ndarray, target: np.ndarray, *,
             allow_scale: bool = False,
             max_iterations: int = 60,
             coarse_points: int = 2_000) -> tuple[np.ndarray, float]:
    """Registriert source auf target und gibt (4x4 Transformation, Massstab).

    ICP ist nur lokal konvergent, deshalb werden mehrere Startlagen probiert:
    die Hauptachsen beider Punktwolken, jeweils mit allen Vorzeichen
    kombinationen. Ohne das landet ICP bei annaehernd symmetrischen
    Objekten regelmaessig in einer um 90 oder 180 Grad verdrehten Lage.

    Zweistufig, weil ICP ueber alle Punkte je Startlage teuer ist: erst
    grob auf einer Stichprobe, dann nur die beste Lage auf dem vollen Satz.
    """
    rng = np.random.default_rng(0)
    n = min(coarse_points, len(source), len(target))
    src_c = source[rng.choice(len(source), n, replace=False)]
    dst_c = target[rng.choice(len(target), n, replace=False)]

    src_frame = _principal_frame(source)
    dst_frame = np.linalg.inv(_principal_frame(target))

    best_initial, best_cost = None, np.inf
    for sx in (1, -1):
        for sy in (1, -1):
            flip = np.diag([sx, sy, sx * sy, 1.0]).astype(np.float64)
            initial = dst_frame @ flip @ src_frame
            transform, _, cost = trimesh.registration.icp(
                src_c, dst_c,
                initial=initial,
                scale=allow_scale,
                max_iterations=20,
            )
            if cost < best_cost:
                best_initial, best_cost = transform, cost

    transform, _, _ = trimesh.registration.icp(
        source, target,
        initial=best_initial,
        scale=allow_scale,
        max_iterations=max_iterations,
    )

    # det der Transformation ist der Faktor, mit dem source verkleinert
    # wird. Berichtet wird der Kehrwert, also "um wieviel ist die
    # Rekonstruktion zu gross gegenueber der Referenz".
    applied = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))
    return transform, 1.0 / applied


def apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return trimesh.transform_points(points, transform)


# --------------------------------------------------------------------------
# Distanzmetriken
# --------------------------------------------------------------------------

def _nn_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Abstand jedes Punktes aus a zum naechsten Punkt in b."""
    return cKDTree(b).query(a, k=1)[0]


@dataclass
class GeometryMetrics:
    chamfer_mean: float        # symmetrisch gemittelt, mm
    chamfer_rms: float         # symmetrisch, quadratisch gemittelt, mm
    hausdorff: float           # Maximum beider Richtungen, mm
    hausdorff_p95: float       # 95. Perzentil, robust gegen Ausreisser
    completeness: float        # Referenz -> Rekonstruktion, mm
    accuracy: float            # Rekonstruktion -> Referenz, mm
    scale_factor: float        # >1 heisst Rekonstruktion zu gross, 1.0 bei starr
    bbox_error: float          # groesste Kantenabweichung der Bounding Box, mm

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compare(recon: trimesh.Trimesh, reference: trimesh.Trimesh, *,
            allow_scale: bool = False,
            samples: int = DEFAULT_SAMPLES,
            seed: int = 0) -> GeometryMetrics:
    """Vergleicht ein rekonstruiertes Mesh mit der Referenz.

    Alle Laengen in der Einheit der Referenz, bei uns Millimeter.
    """
    src = sample_surface(recon, samples, seed)
    dst = sample_surface(reference, samples, seed)

    transform, scale = register(src, dst, allow_scale=allow_scale)
    src = apply(src, transform)

    # accuracy: wie weit liegt die Rekonstruktion neben der Referenz
    # completeness: wie gut ist die Referenz ueberhaupt abgedeckt
    d_src = _nn_distances(src, dst)
    d_dst = _nn_distances(dst, src)

    both = np.concatenate([d_src, d_dst])
    recon_bbox = np.sort(trimesh.PointCloud(src).extents)[::-1]
    ref_bbox = np.sort(reference.extents)[::-1]

    return GeometryMetrics(
        chamfer_mean=float(0.5 * (d_src.mean() + d_dst.mean())),
        chamfer_rms=float(np.sqrt(0.5 * ((d_src**2).mean() + (d_dst**2).mean()))),
        hausdorff=float(both.max()),
        hausdorff_p95=float(np.percentile(both, 95)),
        completeness=float(d_dst.mean()),
        accuracy=float(d_src.mean()),
        scale_factor=scale,
        bbox_error=float(np.abs(recon_bbox - ref_bbox).max()),
    )


# --------------------------------------------------------------------------
# Mesh Qualitaet, unabhaengig von der Referenz
# --------------------------------------------------------------------------

@dataclass
class MeshQuality:
    n_faces: int
    n_vertices: int
    watertight: bool
    winding_consistent: bool
    n_components: int
    n_nonmanifold_edges: int
    volume_cm3: float

    def as_dict(self) -> dict:
        return asdict(self)


def quality(mesh: trimesh.Trimesh) -> MeshQuality:
    edges = mesh.edges_sorted.reshape((-1, 2))
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return MeshQuality(
        n_faces=len(mesh.faces),
        n_vertices=len(mesh.vertices),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        n_components=len(mesh.split(only_watertight=False)),
        n_nonmanifold_edges=int((counts != 2).sum()),
        volume_cm3=float(mesh.volume / 1000.0) if mesh.is_watertight else float("nan"),
    )
