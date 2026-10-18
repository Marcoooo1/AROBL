#!/usr/bin/env python
"""Korrelationsanalyse zwischen Rendering Qualitaet und Geometrie (Teil A).

Fuehrt results/metrics.csv und results/geometry.csv ueber die Spalte exp
zusammen und beantwortet die Kernfrage: sagt die Rendering Qualitaet die
geometrische Genauigkeit vorher.

Methodik und warum so:

Rangkorrelation nach Spearman, nicht Pearson. Wir behaupten keinen linearen
Zusammenhang, sondern fragen, ob eine bessere Rendering Qualitaet mit besserer
Geometrie einhergeht. Dafuer genuegt die Monotonie.

Konfidenzintervall per Bootstrap auf Szenenebene, nicht auf Laufebene. Die
Laeufe einer Szene teilen Objekt, Beleuchtung und Kameratrajektorie, sie sind
also nicht unabhaengig. Ein Bootstrap, der einzelne Laeufe zieht, tut so als
waeren es 75 unabhaengige Beobachtungen und liefert zu enge Intervalle. Der
Szenen Bootstrap zieht ganze Szenen mit Zuruecklegen und ist die ehrlichere
Variante. Beide werden berichtet, damit der Unterschied sichtbar bleibt.

Zusaetzlich die Korrelation je Szene einzeln. Ueber alle Szenen gepoolt kann
ein Zusammenhang allein daher kommen, dass manche Objekte leichter sind als
andere. Bleibt der Zusammenhang auch innerhalb jeder Szene bestehen, ist er
belastbarer.

Vorzeichen: chamfer, accuracy und completeness sind Fehlermasse, da ist klein
besser. psnr, ssim und f_score sind Guetemasse, da ist gross besser. lpips ist
wieder ein Fehlermass. Die erwartete Richtung steht in der Spalte soll und
wird mit ausgegeben, damit niemand ein Vorzeichen falsch liest.

Aufruf:
    python scripts/analyze.py
    python scripts/analyze.py --min-n 10        # Versagensbereich ausschliessen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS_CSV = Path("results/metrics.csv")
GEOM_CSV = Path("results/geometry.csv")
OUT_DIR = Path("results/analysis")

RENDER = ["psnr", "ssim", "lpips"]
GEOM = ["chamfer", "accuracy", "completeness", "f_score"]

# Richtung, die unsere These erwarten liesse, wenn Rendering die Geometrie
# vorhersagt: besseres Bild geht mit besserer Geometrie einher.
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False,
                    "chamfer": False, "accuracy": False,
                    "completeness": False, "f_score": True}


def expected_sign(render: str, geom: str) -> int:
    """+1 wenn die These ein positives rho erwarten laesst, sonst -1."""
    return 1 if HIGHER_IS_BETTER[render] == HIGHER_IS_BETTER[geom] else -1


def load(min_n: int, metrics_csv: Path, geom_csv: Path) -> pd.DataFrame:
    if not metrics_csv.exists() or not geom_csv.exists():
        raise SystemExit(f"{metrics_csv} oder {geom_csv} fehlt. "
                         "Erst run_experiments.py und eval_geometry.py laufen lassen.")
    m = pd.read_csv(metrics_csv)
    g = pd.read_csv(geom_csv)
    m = m[m["status"] == "ok"]
    g = g[g["status"] == "ok"]

    # metrics.csv haelt leere Platzhalterspalten fuer die Geometrie bereit.
    # Die muessen vor dem Zusammenfuehren weg, sonst entstehen chamfer_x und
    # chamfer_y und die Geometriespalten fehlen danach stillschweigend.
    keep_from_m = {"exp", "scene", "n", "seed", "status"}
    overlap = [c for c in g.columns if c in m.columns and c not in keep_from_m]
    m = m.drop(columns=overlap)

    dup = [c for c in ("scene", "n", "seed", "status") if c in g.columns]
    df = m.merge(g.drop(columns=dup), on="exp", how="inner", validate="one_to_one")

    # Laut und frueh scheitern statt leise Spalten zu verlieren.
    missing = [c for c in GEOM if c not in df.columns]
    if missing:
        raise SystemExit(f"nach dem Zusammenfuehren fehlen Spalten: {missing}. "
                         f"geometry.csv pruefen.")
    df = df[df["n"] >= min_n]
    for c in RENDER + GEOM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def bootstrap_ci(df: pd.DataFrame, a: str, b: str, *, by_scene: bool,
                 reps: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Perzentil Bootstrap. by_scene zieht ganze Szenen, sonst einzelne Laeufe."""
    rng = np.random.default_rng(seed)
    rhos = []
    if by_scene:
        scenes = df["scene"].unique()
        if len(scenes) < 2:
            return float("nan"), float("nan")
        groups = [df[df["scene"] == s] for s in scenes]
        for _ in range(reps):
            pick = rng.integers(0, len(groups), len(groups))
            sample = pd.concat([groups[i] for i in pick])
            r, _ = spearman(sample[a].to_numpy(), sample[b].to_numpy())
            if not np.isnan(r):
                rhos.append(r)
    else:
        x, y = df[a].to_numpy(), df[b].to_numpy()
        if len(x) < 3:
            return float("nan"), float("nan")
        for _ in range(reps):
            idx = rng.integers(0, len(x), len(x))
            r, _ = spearman(x[idx], y[idx])
            if not np.isnan(r):
                rhos.append(r)
    if len(rhos) < reps // 10:
        return float("nan"), float("nan")
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def per_scene(df: pd.DataFrame, a: str, b: str) -> list[float]:
    out = []
    for s in sorted(df["scene"].unique()):
        sub = df[df["scene"] == s]
        r, _ = spearman(sub[a].to_numpy(), sub[b].to_numpy())
        if not np.isnan(r):
            out.append(r)
    return out


def analyse(df: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    for a in RENDER:
        for b in GEOM:
            if a not in df.columns or b not in df.columns:
                continue
            sub = df[[a, b, "scene"]].dropna()
            rho, p = spearman(sub[a].to_numpy(), sub[b].to_numpy())
            lo_s, hi_s = bootstrap_ci(sub, a, b, by_scene=True, reps=reps)
            lo_r, hi_r = bootstrap_ci(sub, a, b, by_scene=False, reps=reps)
            scene_rhos = per_scene(sub, a, b)
            rows.append({
                "render": a, "geometry": b, "n_runs": len(sub),
                "rho": rho, "p": p,
                "ci_lo_scene": lo_s, "ci_hi_scene": hi_s,
                "ci_lo_run": lo_r, "ci_hi_run": hi_r,
                "rho_scene_median": float(np.median(scene_rhos)) if scene_rhos else float("nan"),
                "rho_scene_min": float(np.min(scene_rhos)) if scene_rhos else float("nan"),
                "rho_scene_max": float(np.max(scene_rhos)) if scene_rhos else float("nan"),
                "n_scenes": len(scene_rhos),
                "expected_sign": expected_sign(a, b),
            })
    return pd.DataFrame(rows)


def by_image_count(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in RENDER + GEOM if c in df.columns]
    return df.groupby("n")[cols].agg(["mean", "std", "count"])


def fmt(v: float, nd: int = 3) -> str:
    return "  n/a " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:>6.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=0,
                    help="nur Laeufe ab dieser Bildanzahl")
    ap.add_argument("--reps", type=int, default=10_000)
    ap.add_argument("--metrics", type=Path, default=METRICS_CSV)
    ap.add_argument("--geometry", type=Path, default=GEOM_CSV)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    df = load(args.min_n, args.metrics, args.geometry)
    n_scenes = df["scene"].nunique()
    print(f"{len(df)} Laeufe aus {n_scenes} Szene(n), "
          f"Bildanzahlen {sorted(int(v) for v in df['n'].unique())}")

    if len(df) < 10:
        print("\nACHTUNG: zu wenige Laeufe fuer eine belastbare Korrelation.")
        print("Die Zahlen unten sind ein Funktionstest der Auswertung und")
        print("kein Ergebnis. Erst die vollstaendige Messreihe auswerten.\n")

    print("\nMittelwerte je Bildanzahl")
    print(by_image_count(df).round(4).to_string())

    res = analyse(df, args.reps)
    print("\nRangkorrelation nach Spearman")
    print("  erwartet ist das Vorzeichen in der Spalte soll, wenn Rendering")
    print("  die Geometrie vorhersagt\n")
    head = (f"  {'render':<6} {'geometrie':<13} {'rho':>6} {'p':>8} "
            f"{'CI Szene':>16} {'CI Lauf':>16} {'je Szene':>16} {'soll':>5}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for _, r in res.iterrows():
        ci_s = f"[{fmt(r.ci_lo_scene,2).strip()}, {fmt(r.ci_hi_scene,2).strip()}]"
        ci_r = f"[{fmt(r.ci_lo_run,2).strip()}, {fmt(r.ci_hi_run,2).strip()}]"
        sc = (f"{fmt(r.rho_scene_min,2).strip()}..{fmt(r.rho_scene_max,2).strip()}"
              if r.n_scenes else "n/a")
        print(f"  {r.render:<6} {r.geometry:<13} {fmt(r.rho,3)} {fmt(r.p,4)} "
              f"{ci_s:>16} {ci_r:>16} {sc:>16} {r.expected_sign:>+5d}")

    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "correlations.csv", index=False)
    df.to_csv(args.out / "merged.csv", index=False)
    print(f"\ngeschrieben: {args.out/'correlations.csv'} und {args.out/'merged.csv'}")

    print("\nLesehilfe")
    print("  rho nahe 0        Rendering sagt die Geometrie nicht vorher")
    print("  rho mit dem Vorzeichen aus soll und CI, das 0 nicht enthaelt")
    print("                    Zusammenhang vorhanden")
    print("  CI Szene ist breiter als CI Lauf, weil Laeufe derselben Szene")
    print("  nicht unabhaengig sind. Fuer das Paper gilt CI Szene.")
    print("  je Szene zeigt die Spannweite der Einzelszenen. Weit auseinander")
    print("  heisst, der gepoolte Wert ist nicht die ganze Geschichte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
