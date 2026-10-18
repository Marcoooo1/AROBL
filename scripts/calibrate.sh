#!/usr/bin/env bash
# One full-length run on the heaviest config to measure wall-clock time.
# Decides whether 75 runs (5 scenes x 3 seeds) are feasible or we cut to 3x2.
set -euo pipefail

EXP=data/dtu/experiments/scan24_n40_s0
OUT=outputs/calib
NAME=calib_s24n40s0
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

echo "=== TRAINING (30000 steps) ==="
t0=$(date +%s)
ns-train nerfacto \
  --max-num-iterations 30000 \
  --vis tensorboard \
  --output-dir "$OUT" \
  --experiment-name "$NAME" \
  --pipeline.model.predict-normals True \
  --pipeline.model.background-color black \
  --pipeline.model.camera-optimizer.mode off \
  --pipeline.model.use-appearance-embedding False \
  --pipeline.datamanager.pixel-sampler.ignore-mask True \
  --data "$EXP" \
  nerfstudio-data \
    --orientation-method none --center-method none \
    --auto-scale-poses False --mask-color 0 0 0
t1=$(date +%s)

RUN=$(ls -td "$OUT/$NAME/nerfacto/"*/ | head -1)
echo "config at: $RUN"

echo "=== EXPORT (poisson) ==="
t2=$(date +%s)
ns-export poisson \
  --load-config "${RUN}config.yml" \
  --output-dir "$OUT/mesh" \
  --texture-method point_cloud \
  --num-points 1000000
t3=$(date +%s)

echo "=== EVAL (render metrics) ==="
t4=$(date +%s)
ns-eval --load-config "${RUN}config.yml" --output-path /tmp/calib_eval.json
t5=$(date +%s)

echo
echo "================  CALIBRATION  ================"
echo "training : $((t1-t0)) s"
echo "export   : $((t3-t2)) s"
echo "eval     : $((t5-t4)) s"
echo "per run  : $((t1-t0 + t3-t2 + t5-t4)) s"
echo "==============================================="
python - <<'PY'
import json
r = json.load(open("/tmp/calib_eval.json"))["results"]
print(f"PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.3f}  LPIPS {r['lpips']:.3f}")
PY
