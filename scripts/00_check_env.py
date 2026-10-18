#!/usr/bin/env python
"""Prueft die Arbeitsumgebung und nennt bei jedem Fehlschlag die Ursache."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

OK, WARN, FAIL = "\033[92mOK  \033[0m", "\033[93mWARN\033[0m", "\033[91mFAIL\033[0m"


@dataclass
class Result:
    level: str
    name: str
    detail: str
    hint: str = ""


results: list[Result] = []


def record(level, name, detail, hint=""):
    results.append(Result(level, name, detail, hint))


def check_torch_and_gpu():
    try:
        import torch
    except ImportError:
        record(FAIL, "PyTorch", "nicht installiert", "bash scripts/install.sh")
        return

    record(OK, "PyTorch", f"{torch.__version__}, gebaut gegen CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        record(FAIL, "CUDA verfuegbar", "is_available() ist False",
               "nvidia-smi in WSL pruefen. Treiber gehoert nach Windows, nicht nach WSL.")
        return

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    record(OK, "GPU", f"{torch.cuda.get_device_name(0)} ({arch})")

    arch_list = torch.cuda.get_arch_list()
    if arch in arch_list:
        record(OK, "Kernel fuer GPU", f"{arch} enthalten")
    else:
        record(FAIL, "Kernel fuer GPU", f"{arch} fehlt, vorhanden: {', '.join(arch_list)}",
               "Falscher Wheel Index. PyTorch mit --index-url .../cu128 neu installieren.")

    try:  # rechnet wirklich auf der GPU statt nur Metadaten zu lesen
        a = torch.randn(512, 512, device="cuda")
        (a @ a).sum().item()
        torch.cuda.synchronize()
        record(OK, "GPU Rechentest", "erfolgreich")
    except Exception as exc:
        record(FAIL, "GPU Rechentest", str(exc)[:160], "Treiber und CUDA passen nicht zusammen.")

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    record(OK if total >= 7.5 else WARN, "VRAM", f"{total:.1f} GB",
           "" if total >= 7.5 else "Nerfacto mit reduzierter Aufloesung trainieren.")


def check_binaries():
    if shutil.which("nvcc"):
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        v = next((t for t in out.split() if t.startswith("V1")), "unbekannt")
        ok = v.startswith("V12.") and float(v[1:5]) >= 12.8
        record(OK if ok else WARN, "nvcc", v,
               "" if ok else "Blackwell braucht CUDA 12.8. Nur fuer eigene CUDA Builds relevant.")
    else:
        record(WARN, "nvcc", "nicht im PATH", "CUDA_HOME und PATH setzen.")

    for binary, hint in (("colmap", "sudo apt install -y colmap"),
                         ("ns-train", "nerfstudio fehlt oder falsche Umgebung aktiv")):
        if shutil.which(binary):
            record(OK, binary, "verfuegbar")
        else:
            record(FAIL, binary, "nicht gefunden", hint)


def check_imports():
    modules = {
        "nerfstudio": "pip install nerfstudio",
        "transformers": "liefert Grounding DINO",
        "sam2": "pip install git+https://github.com/facebookresearch/sam2.git",
        "open3d": "pip install open3d",
        "trimesh": "pip install trimesh",
        "pymeshlab": "pip install pymeshlab",
        "skimage": "scikit-image, liefert Marching Cubes",
        "scipy": "pip install scipy",
        "pandas": "pip install pandas",
        "yaml": "pip install pyyaml",
    }
    for mod, hint in modules.items():
        try:
            m = importlib.import_module(mod)
            record(OK, mod, getattr(m, "__version__", "importierbar"))
        except Exception as exc:
            record(FAIL, mod, str(exc)[:90], hint)

    try:
        importlib.import_module("tinycudann")
        record(OK, "tiny-cuda-nn", "installiert, implementation=tcnn moeglich")
    except ImportError:
        record(WARN, "tiny-cuda-nn", "nicht installiert",
               "So geplant. Training mit --pipeline.model.implementation torch.")


def check_filesystem():
    cwd = os.path.realpath(os.getcwd())
    if cwd.startswith("/mnt/"):
        record(WARN, "Arbeitsverzeichnis", cwd,
               "Liegt im Windows Dateisystem. Nach ~/projects/ verschieben.")
    else:
        record(OK, "Arbeitsverzeichnis", cwd)


def main() -> int:
    v = sys.version_info
    record(OK if v >= (3, 10) else FAIL, "Python", f"{v.major}.{v.minor}.{v.micro}")
    check_torch_and_gpu()
    check_binaries()
    check_imports()
    check_filesystem()

    print("\nUmgebungspruefung\n" + "=" * 70)
    width = max(len(r.name) for r in results) + 2
    for r in results:
        print(f"{r.level}  {r.name:<{width}} {r.detail}")
        if r.hint:
            print(f"      {'':<{width}} -> {r.hint}")

    fails = sum(r.level == FAIL for r in results)
    warns = sum(r.level == WARN for r in results)
    print("=" * 70)
    print(f"{len(results) - fails - warns} ok, {warns} Warnungen, {fails} Fehler\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
