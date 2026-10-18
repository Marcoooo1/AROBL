#!/usr/bin/env python
"""Patch nerfstudio checkpoint loading for torch >= 2.6.

Background (verified on this machine, 2026-10-16):
torch 2.6 changed the default of `weights_only` in `torch.load` from False to
True. Nerfstudio checkpoints contain numpy scalars, so `ns-eval` and
`ns-export` fail with:

    _pickle.UnpicklingError: Weights only load failed.
    Unsupported global: GLOBAL numpy.core.multiarray.scalar

This script adds `weights_only=False` to every `torch.load` call inside the
installed nerfstudio package. It is idempotent and keeps a .bak copy of every
file it touches.

Security note: weights_only=False allows arbitrary code execution while
unpickling. That is acceptable here because the checkpoints are produced by
this machine. Never apply this to checkpoints from an untrusted source.

Usage:
    python env/patch_torch_load.py --dry-run
    python env/patch_torch_load.py
    python env/patch_torch_load.py --restore
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

CALL = re.compile(r"torch\.load\(")


def find_calls(text: str) -> list[tuple[int, int]]:
    """Return (start, end) index pairs of complete torch.load(...) calls."""
    spans: list[tuple[int, int]] = []
    for match in CALL.finditer(text):
        depth, i = 0, match.end() - 1
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), i + 1))
                    break
            i += 1
    return spans


def patch_text(text: str) -> tuple[str, int]:
    """Add weights_only=False to torch.load calls that lack it."""
    out, last, changed = [], 0, 0
    for start, end in find_calls(text):
        call = text[start:end]
        if "weights_only" in call:
            continue
        inner = call[len("torch.load(") : -1]
        # A trailing comma, possibly followed by whitespace or a newline, would
        # otherwise produce ", , weights_only=False)". Keep the indentation of
        # the closing parenthesis intact.
        stripped = inner.rstrip()
        tail = inner[len(stripped):]
        if stripped.endswith(","):
            stripped = stripped[:-1]
        new_call = f"torch.load({stripped}, weights_only=False{tail})"
        out.append(text[last:start])
        out.append(new_call)
        last = end
        changed += 1
    out.append(text[last:])
    return "".join(out), changed


def package_root() -> Path:
    try:
        import nerfstudio
    except ImportError:
        sys.exit("nerfstudio is not installed in the active environment.")
    return Path(nerfstudio.__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true",
                    help="restore every .bak file and exit")
    args = ap.parse_args()

    root = package_root()
    print(f"nerfstudio at {root}\n")

    if args.restore:
        n = 0
        for bak in root.rglob("*.py.bak"):
            shutil.copy2(bak, bak.with_suffix(""))
            bak.unlink()
            print(f"  restored {bak.with_suffix('').relative_to(root)}")
            n += 1
        print(f"\n{n} file(s) restored\n")
        return 0

    total_files, total_calls = 0, 0
    for path in sorted(root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "torch.load(" not in text:
            continue

        new_text, changed = patch_text(text)
        if changed == 0:
            continue

        rel = path.relative_to(root)
        print(f"  {rel}: {changed} call(s)")
        total_files += 1
        total_calls += changed

        if not args.dry_run:
            backup = path.with_suffix(".py.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(new_text, encoding="utf-8")

    verb = "would patch" if args.dry_run else "patched"
    print(f"\n{verb} {total_calls} call(s) in {total_files} file(s)\n")
    if args.dry_run:
        print("Run without --dry-run to apply.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
