#!/usr/bin/env python
"""Convert a NeuS/IDR-format DTU scene to a Nerfstudio transforms.json.

Input layout (per scene):
    <scene>/cameras_sphere.npz   world_mat_i, scale_mat_i, camera_mat_i
    <scene>/image/000000.png ...  six-digit names
    <scene>/mask/000.png ...      three-digit names

Files are addressed by index, not by globbing, so a stray subfolder or a
second mask variant inside mask/ (scan55 has one) cannot be picked up by
mistake. For view i the image is image/{i:06d}.png and the mask mask/{i:03d}.png.

Output (at the chosen --downscale D):
    <out>/transforms.json         Nerfstudio, normalized frame (unit sphere)
    <out>/images/ , <out>/masks/  resized and renamed to {i:04d}.png
    <out>/scale_mat.npy           back-transform to DTU millimetres

Downscaling is done here, not via Nerfstudio's images_N folders. Images use
area averaging, masks nearest neighbour so they stay binary, intrinsics /= D.

Conventions, verified against a synthetic scene with known ground truth:
    P = (world_mat @ scale_mat)[:3, :4]
    K, R, t = cv2.decomposeProjectionMatrix(P)        # OpenCV frame
    c2w = [R^T | -R^T t];  c2w_gl = c2w @ diag(1,-1,-1,1)   # -> OpenGL

scale_mat maps normalized -> DTU mm: v_dtu = S[:3,:3] @ v_norm + S[:3,3].
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def load_K_pose_from_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
    K = K / K[2, 2]
    c2w = np.eye(4)
    c2w[:3, :3] = R.transpose()
    c2w[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, c2w


CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def convert_scene(scene_dir: Path, out_dir: Path, *,
                  downscale: int = 2, check: bool = True) -> dict:
    cam = np.load(scene_dir / "cameras_sphere.npz")
    n = len([k for k in cam.files if re.match(r"world_mat_\d+$", k)])

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    first = scene_dir / "image" / f"{0:06d}.png"
    if not first.exists():
        raise FileNotFoundError(f"{scene_dir.name}: missing {first}")
    H, W = cv2.imread(str(first)).shape[:2]
    w, h = W // downscale, H // downscale

    frames = []
    inside, centre_dist = [], []
    K_ref = None
    scale_mat = cam["scale_mat_0"]

    for i in range(n):
        img_path = scene_dir / "image" / f"{i:06d}.png"
        msk_path = scene_dir / "mask" / f"{i:03d}.png"
        if not img_path.exists() or not msk_path.exists():
            raise FileNotFoundError(
                f"{scene_dir.name}: view {i} missing "
                f"({img_path.name} / {msk_path.name})"
            )

        P = (cam[f"world_mat_{i}"] @ cam[f"scale_mat_{i}"])[:3, :4]
        K, c2w = load_K_pose_from_P(P)
        if K_ref is None:
            K_ref = K / downscale
            K_ref[2, 2] = 1.0

        if check:
            uvw = P @ np.array([0.0, 0.0, 0.0, 1.0])
            d = uvw[2]
            u, v = uvw[0] / d, uvw[1] / d
            inside.append(d > 0 and 0 <= u < W and 0 <= v < H)
            centre_dist.append(float(np.hypot(u - W / 2, v - H / 2)))

        img = cv2.resize(cv2.imread(str(img_path)), (w, h),
                         interpolation=cv2.INTER_AREA)
        msk = cv2.resize(cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE),
                         (w, h), interpolation=cv2.INTER_NEAREST)
        name = f"{i:04d}.png"
        cv2.imwrite(str(out_dir / "images" / name), img)
        cv2.imwrite(str(out_dir / "masks" / name), msk)

        frames.append({
            "file_path": f"images/{name}",
            "mask_path": f"masks/{name}",
            "transform_matrix": (c2w @ CV_TO_GL).tolist(),
        })

    transforms = {
        "w": int(w), "h": int(h),
        "fl_x": float(K_ref[0, 0]), "fl_y": float(K_ref[1, 1]),
        "cx": float(K_ref[0, 2]), "cy": float(K_ref[1, 2]),
        "camera_model": "OPENCV",
        "frames": frames,
    }
    (out_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))
    np.save(out_dir / "scale_mat.npy", scale_mat)

    info = {
        "scene": scene_dir.name, "n_views": n, "downscale": downscale,
        "resolution": (w, h), "fl_x": float(K_ref[0, 0]),
        "sphere_radius_mm": float(scale_mat[0, 0]),
    }
    if check:
        info["origin_inside_frac"] = float(np.mean(inside))
        info["origin_centre_dist_median_px"] = float(np.median(centre_dist))
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    for k, v in convert_scene(args.scene_dir, args.out_dir,
                              downscale=args.downscale,
                              check=not args.no_check).items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
