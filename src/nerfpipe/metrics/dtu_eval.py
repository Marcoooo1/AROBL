"""DTU Geometrie Auswertung fuer Teil A.

Folgt dem offiziellen DTU MVS Evaluationsprotokoll (Matlab Referenz und
Port DTUeval-python). Ablauf:

    1. Rekonstruiertes Mesh in DTU Weltkoordinaten (Millimeter) bringen.
       Das macht nicht dieses Modul, sondern scripts/eval_geometry.py, weil
       dafuer die scale_mat der Szene und die dataparser_transforms von
       Nerfstudio noetig sind.
    2. Mesh dicht abtasten und per Voxelgitter auf einen festen Mindest
       abstand ausduennen (downsample_density, Standard 0.2 mm).
    3. Punkte ausserhalb der beobachteten Region verwerfen (ObsMask).
    4. accuracy  d2s  Rekonstruktion -> Grundwahrheit, mittlere Distanz.
       completeness  s2d  Grundwahrheit -> Rekonstruktion, mittlere Distanz.
       Die Grundwahrheit wird zuvor an der Bodenebene (Plane) beschnitten.
       Distanzen ueber max_dist (Standard 20 mm) gehen nicht in den
       Mittelwert, das ist die Ausreisserunterdrueckung des Protokolls.
    5. chamfer  overall  Mittel aus accuracy und completeness.
    6. F-Score bei einer Schwelle (Standard 2 mm) aus precision und recall.

Abweichung vom Original, transparent benannt: das Original duennt mit einer
gierigen Radius Nachbarschaft aus, hier ein Voxelgitter gleicher Kantenlaenge.
Beide erzwingen denselben Mindestabstand. Fuer den Vergleich der Laeufe
untereinander ist nur wichtig, dass das Protokoll fuer alle Laeufe identisch
ist, und das ist es.

Dieses Modul kennt nur Geometrie. Rendering Metriken gehoeren nicht hierher.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.io import loadmat
from scipy.spatial import cKDTree

DOWNSAMPLE_DENSITY = 0.2   # mm, Kantenlaenge des Ausduennungsgitters
MAX_DIST = 20.0            # mm, Distanzen darueber zaehlen nicht zum Mittel
PATCH = 60                 # mm, Rand um die Bounding Box fuer die ObsMask
FSCORE_THRESH = 2.0        # mm, Schwelle fuer precision und recall


# --------------------------------------------------------------------------
# Grundwahrheit laden
# --------------------------------------------------------------------------

@dataclass
class DTUGroundTruth:
    """Alles aus dem offiziellen DTU SampleSet fuer eine Szene."""
    points: np.ndarray         # Nx3, Punktwolke des Laserscans in mm
    obs_mask: np.ndarray       # 3D bool Gitter der beobachteten Region
    bb: np.ndarray             # 2x3, Minimum und Maximum des Gitters in mm
    res: float                 # Kantenlaenge des ObsMask Gitters in mm
    plane: np.ndarray          # 4, Bodenebene zum Beschneiden der Grundwahrheit


def load_ground_truth(dataset_dir: str | Path, scan: int) -> DTUGroundTruth:
    """Laedt Punktwolke, ObsMask und Bodenebene fuer eine DTU Szene.

    Erwartete Ablage unter dataset_dir, so wie das offizielle SampleSet und
    die Points Archive sie ablegen:
        Points/stl/stl{scan:03d}_total.ply
        ObsMask/ObsMask{scan}_10.mat   mit ObsMask, BB, Res
        ObsMask/Plane{scan}.mat        mit P
    """
    dataset_dir = Path(dataset_dir)
    stl_path = dataset_dir / "Points" / "stl" / f"stl{scan:03d}_total.ply"
    obs_path = dataset_dir / "ObsMask" / f"ObsMask{scan}_10.mat"
    plane_path = dataset_dir / "ObsMask" / f"Plane{scan}.mat"
    for p in (stl_path, obs_path, plane_path):
        if not p.exists():
            raise FileNotFoundError(
                f"DTU Grundwahrheit fehlt: {p}. Das offizielle SampleSet und "
                f"die Points liegen noch nicht unter {dataset_dir}."
            )

    stl = np.asarray(trimesh.load(stl_path).vertices, dtype=np.float64)

    obs = loadmat(str(obs_path))
    obs_mask = obs["ObsMask"].astype(bool)
    bb = obs["BB"].astype(np.float64)
    res = float(np.asarray(obs["Res"]).reshape(-1)[0])

    plane = loadmat(str(plane_path))["P"].astype(np.float64).reshape(-1)
    return DTUGroundTruth(points=stl, obs_mask=obs_mask, bb=bb, res=res, plane=plane)


# --------------------------------------------------------------------------
# Mesh abtasten und ausduennen
# --------------------------------------------------------------------------

def sample_mesh(mesh: trimesh.Trimesh, density: float = DOWNSAMPLE_DENSITY,
                oversample: float = 1.5, cap: int = 4_000_000,
                seed: int = 0) -> np.ndarray:
    """Dichte Oberflaechenabtastung in der Einheit des Mesh (mm).

    Die Punktzahl wird aus der Flaeche geschaetzt, sodass nach dem spaeteren
    Ausduennen jede Zelle des Gitters getroffen ist.
    """
    area = float(mesh.area)
    n = int(min(cap, max(10_000, area / (density ** 2) * oversample)))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def voxel_downsample(points: np.ndarray, voxel: float = DOWNSAMPLE_DENSITY,
                     seed: int = 0) -> np.ndarray:
    """Haelt je belegter Gitterzelle genau einen Punkt, erzwingt so einen
    Mindestabstand in der Groessenordnung voxel."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    pts = points[order]
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


# --------------------------------------------------------------------------
# Beobachtete Region und Bodenebene
# --------------------------------------------------------------------------

def filter_obs_mask(points: np.ndarray, gt: DTUGroundTruth,
                    patch: int = PATCH) -> np.ndarray:
    """Behaelt nur Punkte, die im beobachteten Volumen der Szene liegen.

    Alles ausserhalb hat kein Gegenstueck in der Grundwahrheit und wuerde
    die accuracy verfaelschen. Genau das verwirft das Protokoll hier.
    """
    bb_min, bb_max = gt.bb[0], gt.bb[1]
    inbound = ((points >= bb_min - patch) & (points < bb_max + patch * 2)).all(axis=1)
    pts = points[inbound]
    if len(pts) == 0:
        return pts
    grid = np.around((pts - bb_min) / gt.res).astype(np.int64)
    dims = np.array(gt.obs_mask.shape)
    valid = ((grid >= 0) & (grid < dims)).all(axis=1)
    grid, pts = grid[valid], pts[valid]
    keep = gt.obs_mask[grid[:, 0], grid[:, 1], grid[:, 2]]
    return pts[keep]


def filter_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Schneidet die Grundwahrheit an der Bodenebene ab, behaelt die
    Objektseite. plane ist [a,b,c,d] mit a x + b y + c z + d."""
    hom = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    above = (hom * plane.reshape(1, 4)).sum(axis=1) > 0
    return points[above]


# --------------------------------------------------------------------------
# Metriken
# --------------------------------------------------------------------------

@dataclass
class DTUMetrics:
    accuracy: float            # d2s, Rekonstruktion -> Grundwahrheit, mm
    completeness: float        # s2d, Grundwahrheit -> Rekonstruktion, mm
    chamfer: float             # overall, Mittel aus accuracy und completeness
    f_score: float             # bei fscore_thresh
    precision: float           # Anteil Rekonstruktionspunkte unter Schwelle
    recall: float              # Anteil Grundwahrheitspunkte unter Schwelle
    fscore_thresh: float       # verwendete Schwelle in mm
    n_data: int                # Rekonstruktionspunkte nach ObsMask
    n_gt: int                  # Grundwahrheitspunkte nach Bodenebene

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate(data_points: np.ndarray, gt: DTUGroundTruth, *,
             max_dist: float = MAX_DIST,
             fscore_thresh: float = FSCORE_THRESH) -> DTUMetrics:
    """DTU Metriken fuer eine bereits ausgeduennte, in DTU mm vorliegende
    Rekonstruktionspunktwolke."""
    data_in = filter_obs_mask(data_points, gt)
    gt_above = filter_plane(gt.points, gt.plane)
    if len(data_in) == 0 or len(gt_above) == 0:
        raise ValueError(
            "nach ObsMask oder Bodenebene keine Punkte uebrig. Liegt das Mesh "
            "wirklich in DTU Weltkoordinaten? Transform pruefen."
        )

    # accuracy: jeder Rekonstruktionspunkt zur naechsten Grundwahrheit
    d2s = cKDTree(gt_above).query(data_in, k=1)[0]
    # completeness: jede Grundwahrheit zum naechsten Rekonstruktionspunkt
    s2d = cKDTree(data_in).query(gt_above, k=1)[0]

    acc = float(d2s[d2s < max_dist].mean())
    comp = float(s2d[s2d < max_dist].mean())

    precision = float((d2s < fscore_thresh).mean())
    recall = float((s2d < fscore_thresh).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return DTUMetrics(
        accuracy=acc,
        completeness=comp,
        chamfer=0.5 * (acc + comp),
        f_score=f,
        precision=precision,
        recall=recall,
        fscore_thresh=fscore_thresh,
        n_data=int(len(data_in)),
        n_gt=int(len(gt_above)),
    )


def evaluate_mesh(mesh_dtu_mm: trimesh.Trimesh, gt: DTUGroundTruth, *,
                  density: float = DOWNSAMPLE_DENSITY,
                  max_dist: float = MAX_DIST,
                  fscore_thresh: float = FSCORE_THRESH,
                  seed: int = 0) -> DTUMetrics:
    """Komplettpfad: Mesh (schon in DTU mm) abtasten, ausduennen, auswerten."""
    pts = sample_mesh(mesh_dtu_mm, density=density, seed=seed)
    pts = voxel_downsample(pts, voxel=density, seed=seed)
    return evaluate(pts, gt, max_dist=max_dist, fscore_thresh=fscore_thresh)
