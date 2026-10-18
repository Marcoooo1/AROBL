#!/usr/bin/env python
"""Compute mesh statistics in an isolated, memory-capped subprocess.

Run as:  python meshstat_helper.py <mesh.ply>
Prints a JSON dict to stdout, or exits non-zero on failure (including a
clean MemoryError when the address-space cap is hit on a pathological mesh).
"""
import json, resource, sys

# Cap address space at 4 GiB so a broken mesh raises MemoryError here instead
# of triggering the system OOM killer (which could take the parent down).
LIMIT = 4 * 1024**3
try:
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
except Exception:
    pass

import numpy as np
import trimesh

m = trimesh.load_mesh(sys.argv[1], process=False)
if isinstance(m, trimesh.Scene):
    m = trimesh.util.concatenate(tuple(m.geometry.values()))

n_faces = int(len(m.faces))
n_vertices = int(len(m.vertices))

# Component count and manifoldness are the expensive parts. On a huge or
# degenerate mesh they are not informative anyway, so skip past a cap.
stats = {"n_faces": n_faces, "n_vertices": n_vertices,
         "watertight": "", "n_components": "", "n_nonmanifold": ""}
if n_faces <= 2_000_000:
    m.merge_vertices()
    edges = m.edges_sorted.reshape((-1, 2))
    _, counts = np.unique(edges, axis=0, return_counts=True)
    stats["watertight"] = bool(m.is_watertight)
    stats["n_components"] = len(m.split(only_watertight=False))
    stats["n_nonmanifold"] = int((counts != 2).sum())

print(json.dumps(stats))
