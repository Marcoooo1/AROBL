#!/usr/bin/env python
"""Guard nerfstudio's human_format against non-positive numbers.

Background (verified on this machine, 2026-10-17, WSL):
nerfstudio/utils/printing.py human_format does int(floor(log(num, k))).
On this machine the rays-per-second timing occasionally comes out as 0 or
negative (the same glitch that produces negative num_rays_per_sec in
ns-eval output). log of a non-positive number raises ValueError: math
domain error, which crashes training during stats logging, intermittently.

This inserts a guard that returns the number unformatted when it is <= 0.
Display only, it does not change any metric. Idempotent, keeps a .bak.

Usage:
    python env/patch_printing.py            # apply
    python env/patch_printing.py --restore  # undo
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ANCHOR = "    magnitude = int(floor(log(num, k)))"
GUARD = "    if num <= 0:\n        return f\"{num:.2f} \"\n"


def target() -> Path:
    try:
        import nerfstudio.utils.printing as p
    except ImportError:
        sys.exit("nerfstudio is not installed in the active environment.")
    return Path(p.__file__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    path = target()
    bak = path.with_suffix(".py.bak")
    print(f"target: {path}")

    if args.restore:
        if bak.exists():
            shutil.copy2(bak, path)
            bak.unlink()
            print("restored")
        else:
            print("no backup, nothing to restore")
        return 0

    text = path.read_text(encoding="utf-8")
    if "if num <= 0:" in text:
        print("already patched")
        return 0
    if text.count(ANCHOR) != 1:
        sys.exit(f"anchor found {text.count(ANCHOR)} times, expected 1. Aborting.")

    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(text.replace(ANCHOR, GUARD + ANCHOR), encoding="utf-8")
    print("patched human_format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
