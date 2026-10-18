#!/usr/bin/env bash
# Umgebung fuer WSL2 mit RTX 5070 (Blackwell, sm_120), bewusst ohne tiny-cuda-nn
set -euo pipefail

ENV_NAME="nerf-recon"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

command -v conda >/dev/null || { echo "conda fehlt"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "GPU wird in WSL nicht durchgereicht"; exit 1; }

eval "$(conda shell.bash hook)"
conda env list | grep -qE "^${ENV_NAME}\s" || conda create -y -n "${ENV_NAME}" python=3.11
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

echo "==> PyTorch mit CUDA 12.8"
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"

echo "==> Pruefe ob sm_120 Kernel enthalten sind"
python -c "import sys, torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() ist False'; cap = torch.cuda.get_device_capability(0); arch = 'sm_%d%d' % cap; print(torch.cuda.get_device_name(0), arch, 'torch', torch.__version__); sys.exit(0 if arch in torch.cuda.get_arch_list() else arch + ' fehlt. Falscher Wheel Index.')"

echo "==> Nerfstudio, Grounding DINO, SAM 2"
pip install nerfstudio
pip install "transformers>=4.44" accelerate
pip install "git+https://github.com/facebookresearch/sam2.git"

echo "==> Mesh, Metriken, Auswertung"
pip install open3d trimesh pymeshlab scipy scikit-image numpy-stl
pip install pandas matplotlib pyyaml tqdm rich torchmetrics lpips pytest ruff

pip install -e .
echo "Fertig. Weiter mit: python scripts/00_check_env.py"
