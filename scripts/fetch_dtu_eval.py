#!/usr/bin/env python
"""Holt nur die benoetigten Dateien der DTU Auswertungsreferenz.

Die offiziellen Archive sind je etwa 6.3 GB gross, wir brauchen daraus aber
nur drei Dateien je Szene. Zip Archive haben ihr Inhaltsverzeichnis am Ende,
daher laesst sich ueber HTTP Range Requests zuerst nur dieses Verzeichnis
lesen und danach gezielt einzelne Eintraege holen. Das erledigt remotezip,
die Schnittstelle ist dieselbe wie bei zipfile.ZipFile.

Benoetigt je Szene:
    Points.zip      stl{scan:03d}_total.ply   Laserscan Punktwolke in mm
    SampleSet.zip   ObsMask{scan}_10.mat      beobachtetes Volumen
    SampleSet.zip   Plane{scan}.mat           Bodenebene

Ablage, genau so wie nerfpipe.metrics.dtu_eval sie erwartet:
    <out>/Points/stl/stl{scan:03d}_total.ply
    <out>/ObsMask/ObsMask{scan}_10.mat
    <out>/ObsMask/Plane{scan}.mat

Die Eintraege werden ueber den Dateinamen gesucht, nicht ueber einen erwarteten
Pfad, damit die Verschachtelung im Archiv egal ist. Geschrieben wird immer auf
einen von uns berechneten Pfad und nie auf den Pfad aus dem Archiv.

Aufruf:
    python scripts/fetch_dtu_eval.py --dry-run
    python scripts/fetch_dtu_eval.py
    python scripts/fetch_dtu_eval.py --local-points ~/Downloads/Points.zip \
                                     --local-sampleset ~/Downloads/SampleSet.zip
"""

from __future__ import annotations

import argparse
import posixpath
import sys
import zipfile
from pathlib import Path

POINTS_URL = "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
SAMPLESET_URL = "http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
SCANS = [24, 37, 55, 63, 83]


def wanted(scan: int) -> dict[str, str]:
    """Dateiname im Archiv je Zielpfad relativ zum Ausgabeordner."""
    return {
        f"Points/stl/stl{scan:03d}_total.ply": f"stl{scan:03d}_total.ply",
        f"ObsMask/ObsMask{scan}_10.mat": f"ObsMask{scan}_10.mat",
        f"ObsMask/Plane{scan}.mat": f"Plane{scan}.mat",
    }


def open_archive(source: str, local: bool):
    if local:
        return zipfile.ZipFile(source)
    try:
        from remotezip import RemoteZip
    except ImportError:
        sys.exit("remotezip fehlt. Entweder 'pip install remotezip' oder die "
                 "Archive von Hand laden und --local-points und "
                 "--local-sampleset angeben.")
    return RemoteZip(source)


def build_index(names: list[str]) -> dict[str, str]:
    """Dateiname ohne Pfad auf den vollen Eintrag im Archiv abbilden.

    Mehrdeutige Namen koennen vorkommen, dann gewinnt der kuerzeste Pfad.
    """
    index: dict[str, str] = {}
    for n in names:
        if n.endswith("/"):
            continue
        # Eintraege mit .. oder absolutem Pfad sind in einem sauberen Archiv
        # nicht noetig. Wir schreiben zwar ohnehin auf eigene Pfade, aber ein
        # solcher Eintrag koennte sonst den echten verdraengen.
        parts = n.split("/")
        if n.startswith("/") or ".." in parts:
            continue
        base = posixpath.basename(n)
        if base not in index or len(n) < len(index[base]):
            index[base] = n
    return index


def extract(archive, index: dict[str, str], targets: dict[str, str],
            out: Path, dry_run: bool) -> list[str]:
    """Holt die gewuenschten Eintraege. Gibt die Namen zurueck, die fehlten."""
    missing = []
    for rel_target, basename in targets.items():
        entry = index.get(basename)
        if entry is None:
            missing.append(basename)
            continue
        dest = out / rel_target          # selbst berechnet, nie aus dem Archiv
        if dest.exists():
            print(f"    vorhanden  {rel_target}")
            continue
        if dry_run:
            print(f"    wuerde holen  {rel_target}  (aus {entry})")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with archive.open(entry) as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        tmp.replace(dest)
        print(f"    geholt     {rel_target}  {dest.stat().st_size/1e6:.1f} MB")
    return missing


def process(label: str, source: str, local: bool, scans: list[int],
            keys: list[str], out: Path, dry_run: bool) -> list[str]:
    print(f"\n{label}: {source}")
    archive = open_archive(source, local)
    index = build_index(archive.namelist())
    print(f"  Inhaltsverzeichnis gelesen, {len(index)} Dateien im Archiv")
    missing = []
    for scan in scans:
        print(f"  scan{scan}")
        targets = {k: v for k, v in wanted(scan).items()
                   if any(k.startswith(p) for p in keys)}
        missing += extract(archive, index, targets, out, dry_run)
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", type=int, default=SCANS)
    ap.add_argument("--out", type=Path, default=Path("data/dtu/eval"))
    ap.add_argument("--points-url", default=POINTS_URL)
    ap.add_argument("--sampleset-url", default=SAMPLESET_URL)
    ap.add_argument("--local-points", default=None,
                    help="bereits heruntergeladene Points.zip")
    ap.add_argument("--local-sampleset", default=None,
                    help="bereits heruntergeladene SampleSet.zip")
    ap.add_argument("--dry-run", action="store_true",
                    help="nur das Inhaltsverzeichnis lesen und zeigen, was fehlt")
    args = ap.parse_args()

    missing = []
    missing += process("Points", args.local_points or args.points_url,
                       bool(args.local_points), args.scans, ["Points/"],
                       args.out, args.dry_run)
    missing += process("SampleSet", args.local_sampleset or args.sampleset_url,
                       bool(args.local_sampleset), args.scans, ["ObsMask/"],
                       args.out, args.dry_run)

    if missing:
        print("\nNicht im Archiv gefunden:")
        for m in missing:
            print(f"  {m}")
        print("Dateinamen im Archiv pruefen, die Benennung kann abweichen.")
        return 1

    if args.dry_run:
        print("\nTrockenlauf, nichts geschrieben.")
        return 0

    print("\nPruefung:")
    ok = True
    for scan in args.scans:
        for rel in wanted(scan):
            p = args.out / rel
            good = p.exists() and p.stat().st_size > 0
            ok = ok and good
            print(f"  {'OK  ' if good else 'FEHLT'}  {rel}")
    print("\nvollstaendig" if ok else "\nunvollstaendig")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
