#!/usr/bin/env bash
# Short iteration sweep on one config to find where held-out quality plateaus.
# Same train/test split throughout, only max-num-iterations changes.
set -euo pipefail

EXP=data/dtu/experiments/scan24_n40_s0
OUT=outputs/sweep
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

for STEPS in 3000 7000 12000; do
  echo "=== $STEPS steps ==="
  t0=$(date +%s)
  ns-train nerfacto \
    --max-num-iterations "$STEPS" \
    --vis tensorboard \
    --output-dir "$OUT" \
    --experiment-name "sweep_$STEPS" \
    --pipeline.model.predict-normals True \
    --pipeline.model.background-color black \
    --pipeline.model.camera-optimizer.mode off \
    --pipeline.model.use-appearance-embedding False \
    --pipeline.datamanager.pixel-sampler.ignore-mask True \
    --data "$EXP" \
    nerfstudio-data \
      --orientation-method none --center-method none \
      --auto-scale-poses False --mask-color 0 0 0 \
    > "/tmp/sweep_$STEPS.log" 2>&1
  t1=$(date +%s)
  RUN=$(ls -td "$OUT/sweep_$STEPS/nerfacto/"*/ | head -1)
  ns-eval --load-config "${RUN}config.yml" \
    --output-path "/tmp/sweep_eval_$STEPS.json" > /dev/null 2>&1
  python - "$STEPS" "$((t1-t0))" "/tmp/sweep_eval_$STEPS.json" <<'PY'
import json, sys
steps, secs, path = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(path))["results"]
print(f"  {steps:>6} steps  {secs:>4}s  PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.3f}  LPIPS {r['lpips']:.3f}")
PY
done
echo "done"
