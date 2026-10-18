# NeRF Object Reconstruction

Applied Robot Learning, group 7, FH Technikum Wien.
Marco Bayreder, Lorenz Christian.

## 1. What this repository does

It tests whether the rendering quality of a NeRF predicts the geometric
accuracy of the mesh extracted from it. Rendering quality is measured with
PSNR, SSIM and LPIPS on held-out views. Geometric accuracy is measured with
the DTU evaluation protocol against a laser scan. Image count and seed
produce reconstructions of varying quality, a rank correlation then relates
the two. That correlation is the contribution. The photo to STL pipeline is
secondary and serves as the tool.

The work has two parts:

* **Part A, quantitative.** Runs on the DTU dataset, which ships laser-scanned
  ground truth and calibrated camera poses, so no COLMAP and no own reference
  are needed. This is the main contribution.
* **Part B, qualitative.** A demo on self-captured, 3D-printed objects. Shown
  qualitatively and not measured against the CAD STL.

Everything below is Part A unless stated otherwise.

One run of Part A produces two rows, one in `results/metrics.csv` with the
rendering metrics and one in `results/geometry.csv` with the geometry metrics.
They are joined on the column `exp` by the analysis step. The two tables are
kept apart on purpose so rendering and geometry never get mixed up.

Pipeline per run:

1. convert the DTU scene to Nerfstudio format, keep the scene scale matrix
2. select the training views for this image count and seed
3. train nerfacto for a fixed 12000 iterations
4. evaluate PSNR, SSIM and LPIPS on the fixed held-out test views
5. export a Poisson mesh, limited to the unit sphere
6. map the mesh back to DTU millimetres and crop it
7. evaluate accuracy, completeness, chamfer and F-score against the laser scan

---

## 2. Hardware and prerequisites

Any NVIDIA GPU with compute capability 7.5 or higher. tiny-cuda-nn needs 7.5
as a minimum, below that use the pure PyTorch path.

Developed on two machines:

| | GPU | Compute capability | VRAM |
|---|---|---|---|
| Marco | RTX 5070 Laptop, Blackwell | 12.0 | 8 GB |
| Lorenz | RTX 2080 Ti, Turing | 7.5 | 11 GB |

Blackwell is the reason the standard Nerfstudio install does not work. It
needs CUDA 12.8 and the PyTorch cu128 wheels. The documented CUDA 11.8 path
has no sm_120 kernels and fails. The cu128 wheels also cover 7.5, so both
machines use the same PyTorch.

Host system is Windows 11 with WSL2 and Ubuntu 24.04. About 25 GB of free
disk space for the DTU data, the checkpoints and the exported meshes.

---

## 3. Install

Run the steps in order. Expect 60 to 90 minutes, mostly downloads.

### 3.1 Windows side

Install a current NVIDIA driver under Windows, then in PowerShell check:

```powershell
nvidia-smi
```

It must list your GPU and report CUDA 12.8 or higher. Then:

```powershell
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
```

Never install an NVIDIA driver inside WSL, the Windows driver is passed
through. Always start WSL with `wsl ~` so you land in your Linux home and not
in a Windows system folder, otherwise downloads fail with permission errors.

### 3.2 Ubuntu base and CUDA 12.8

```bash
wsl ~
nvidia-smi                 # must show the GPU inside WSL too
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl unzip ffmpeg colmap

# CUDA toolkit 12.8. Do not use apt install nvidia-cuda-toolkit, it is too old.
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8

echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version             # must say release 12.8
```

### 3.3 Miniconda and the environment

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# accept the channel terms once
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Clone into the Linux filesystem and not under /mnt/c, or training is slow.

```bash
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Marcoooo1/AROBL nerf-object-reconstruction
cd nerf-object-reconstruction
bash scripts/install.sh
pip install opencv-python-headless remotezip
```

`install.sh` creates the conda environment `nerf-recon` with Python 3.11 and
installs PyTorch from the cu128 index, Nerfstudio, Grounding DINO, SAM 2 and
the mesh and metric libraries. It stops if the PyTorch install has no kernels
for your GPU, which is the point where this otherwise fails silently later.

### 3.4 tiny-cuda-nn

Not strictly required but about six times faster than the pure PyTorch path.
The architecture is read from the installed card, so do not hard-code it.

```bash
conda activate nerf-recon
pip install "setuptools<80"
export TCNN_CUDA_ARCHITECTURES=$(python -c "import torch;cc=torch.cuda.get_device_capability(0);print(f'{cc[0]}{cc[1]}')")
echo "building for architecture $TCNN_CUDA_ARCHITECTURES"
export LDFLAGS="-L/usr/lib/wsl/lib -L/usr/local/cuda-12.8/lib64/stubs"
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

`setuptools<80` works around a missing pkg_resources and the LDFLAGS line tells
the linker where the WSL CUDA stub lives. If the build fails, uninstall
tinycudann and continue, everything works without it and only slower.

### 3.5 Patches

Four patches. They modify the installed Nerfstudio package and this
repository's runner, so each person applies them on their own machine. The two
Nerfstudio ones must be re-applied after any `pip install --upgrade nerfstudio`.
All four are idempotent and keep a backup copy.

```bash
python env/patch_torch_load.py      # lets ns-eval and ns-export load checkpoints
python env/patch_printing.py        # stops training crashing on a stats log glitch
python env/patch_export.py          # limits the Poisson export to the unit sphere
python env/patch_limit.py           # adds --exp and --limit to the runner
```

`patch_torch_load` adds `weights_only=False` to Nerfstudio's `torch.load`
calls. That is safe only because the checkpoints are produced on your own
machine. Never apply it to checkpoints from an untrusted source.
`patch_printing` guards a logarithm in the stats printer against a
non-positive timing value. `patch_export` is explained in section 6.

### 3.6 Verify

```bash
conda activate nerf-recon
python scripts/00_check_env.py
```

---

## 4. Data setup

### 4.1 Preprocessed DTU images, masks and poses

Download `data_DTU.zip` from the NeuS data release and extract it so the five
scenes 24, 37, 55, 63 and 83 sit under `data/dtu/neus/dtu_scan<N>/`. Each
scene has `cameras_sphere.npz`, `image/` and `mask/`.

### 4.2 Convert to Nerfstudio format

```bash
for s in 24 37 55 63 83; do
  python scripts/dtu_prepare.py data/dtu/neus/dtu_scan$s data/dtu/prepared/scan$s --downscale 2
done
```

Each scene must report `origin_inside_frac 1.0`, which confirms the camera
conversion. The downscale to 800 by 600 keeps training inside 8 GB. This also
writes `scale_mat.npy` per scene, which the geometry stage needs to map the
mesh back to millimetres. Do not delete it.

### 4.3 Generate the experiment splits

```bash
for s in 24 37 55 63 83; do
  python scripts/dtu_splits.py data/dtu/prepared/scan$s data/dtu/experiments --scene scan$s
done
python scripts/run_experiments.py --dry-run     # must list 75 experiments
```

This writes 75 experiment directories, five scenes times the image counts
5, 10, 20, 30 and 40 times three seeds. The test set of six evenly spaced
views is fixed per scene and never trained on. The training subsets are
nested, so the five views of N = 5 are contained in N = 10 and so on, which
means a difference between image counts is not also a difference in which
views were picked.

These directories use symbolic links into `data/dtu/prepared/`. Copying them
between machines over Windows breaks the links, so regenerate them locally
instead of copying.

### 4.4 Official DTU evaluation reference

Needed for the geometry metrics and for nothing else. The official archives
are 6.3 GB each but only three files per scene are used, so the script reads
the archive directory over HTTP range requests and pulls just those, about
625 MB in total.

```bash
python scripts/fetch_dtu_eval.py --dry-run   # cheap, only reads the listing
python scripts/fetch_dtu_eval.py
```

If the dry run fails the server does not support range requests. Download both
archives by hand and point the script at them:

```bash
python scripts/fetch_dtu_eval.py --local-points ~/Downloads/Points.zip \
                                 --local-sampleset ~/Downloads/SampleSet.zip
```

---

## 5. Running the study

Always do a single run first. It takes about 16 minutes and proves the whole
chain works before you commit to many hours.

```bash
conda activate nerf-recon
python scripts/run_experiments.py --exp scan24_n05_s0
python scripts/eval_geometry.py  --exp scan24_n05_s0 --crop-radius 1.0
```

`metrics.csv` must show status ok with a PSNR value, and `geometry.csv` must
show numbers in chamfer and f_score. Then the full series:

```bash
python scripts/run_experiments.py --scenes scan24     # one scene, 15 runs
python scripts/run_experiments.py                     # all 75
python scripts/eval_geometry.py --all --crop-radius 1.0
```

Each run trains 12000 iterations, evaluates the render metrics on the held-out
views and exports a Poisson mesh. 12000 was chosen from an iteration sweep
where quality stopped improving, 20.8 dB at 3000, 22.6 dB at 7000 and 22.6 dB
at 12000. Fixing it means training time is not a hidden variable between runs.
Budget roughly 11 minutes of training plus 5 minutes of export per run.

The runner is resumable. Completed runs are skipped, so you can stop with
Ctrl+C and re-run to continue. A failed run is written as a row with the
failing stage and does not stop the others. If the GPU disappears the series
stops cleanly instead of falling back to CPU.

```bash
python scripts/run_experiments.py --dry-run           # what would run
python scripts/run_experiments.py --limit 3           # smoke test
python scripts/run_experiments.py --retry-failed      # redo failed rows
column -t -s, results/metrics.csv | cut -c1-140
```

Run the whole series on one machine. Splitting it across two would put the
hardware into the results, either confounded with the scene or hidden inside
the seed spread that we use as the reliability measure.

---

## 6. Geometry stage

`scripts/eval_geometry.py` takes the exported mesh, maps it back into DTU
world coordinates in millimetres and evaluates it against the laser scan.

The mapping is the part where a mistake makes every number meaningless, so it
is done in two steps and checked. First the inverse of the Nerfstudio
dataparser transform, which is written per run next to the checkpoint, giving
the normalized frame in which the object sits inside the unit sphere. Then the
scene's `scale_mat`, giving DTU millimetres. The two are composed into one
matrix and verified against the identity that composing it with the forward
transform reproduces `scale_mat` exactly.

`--crop-radius 1.0` limits the mesh to the unit sphere, which is the region
the DTU normalization is defined for. Use it consistently for every run.

The evaluation follows the official DTU protocol. The mesh is sampled and
thinned to a minimum point spacing of 0.2 mm, points outside the observed
volume are discarded using the ObsMask, the reference is cut at its ground
plane and distances above 20 mm are left out of the means. It reports
accuracy, which is reconstruction to reference, completeness, which is
reference to reconstruction, chamfer as the mean of both, and the F-score at
2 mm from precision and recall. One deviation from the original, stated here
because it belongs in the paper: thinning uses a voxel grid where the original
uses a greedy radius search. Both enforce the same minimum spacing, absolute
values can differ slightly from published DTU numbers, and the protocol is
identical across all our runs, which is what the comparison needs.

### Why the export needs limiting

Without a bounding volume, Poisson lays a surface over the far background
point cloud. Measured on scan24_n05_s0, the mesh reached radius 828 in a frame
where the object sits inside radius 1, had 754153 vertices at 57.7 MB and
contained no vertex at the object at all. Note that the Poisson exporter only
inherits `--use-bounding-box` and `--bounding-box-min/-max` and does not act on
them, the oriented box options are the ones that work. With
`--obb-center 0 0 0 --obb-rotation 0 0 0 --obb-scale 2 2 2`, which is the range
-1 to 1, 88.6 percent of the vertices land inside the unit sphere and the mesh
drops to 162547 vertices. `env/patch_export.py` writes these flags into the
runner. The export becomes slower, about 5 minutes instead of 30 seconds,
because rays that miss the box produce no points.

---

## 7. Analysis

```bash
python scripts/analyze.py
python scripts/analyze.py --min-n 10     # exclude the failure regime
```

It joins the two result tables on `exp` and reports the Spearman rank
correlation between every rendering metric and every geometry metric.

Three things to know when reading the output. Spearman and not Pearson,
because we claim monotonicity and not a linear relation. The confidence
interval is computed twice, once by resampling runs and once by resampling
whole scenes. Runs of one scene share the object, the lighting and the camera
trajectory, so they are not independent and the run-level interval is too
narrow. The scene-level interval is the one to report. The correlation is also
computed per scene, because a pooled relation can come from nothing but some
objects being easier than others, and a relation that survives within each
scene is the stronger claim.

A column `soll` gives the sign the hypothesis would predict, so a sign cannot
be misread. Below ten runs the script prints a warning and treats its own
output as a function test rather than a result. Entries show `n/a` when a
correlation is not computable, for example with fewer than three runs or with
no variation in one of the two columns. That is the intended behaviour, a
number from two points would always be plus or minus one and would mean
nothing.

Output goes to `results/analysis/correlations.csv` and `merged.csv`.

---

## 8. Diagnostics

Use these when a mesh or a number looks wrong, before changing any code.

```bash
python scripts/diag_frame.py              # which coordinate frame is the mesh in
python scripts/diag_transform.py          # transform wrong, or just junk in the mesh
```

`diag_frame.py` is the one to run first. It compares the vertex radii against
the camera positions and the scene scale and reports one of three findings:
A the mesh sits in the normalized frame as it should, B the mesh is already in
DTU millimetres, C the mesh lies far beyond the cameras and is background
junk. A bare bounding box cannot tell these apart, which is why this exists.

---

## 9. Result files

```
results/metrics.csv                one row per run, rendering metrics and run metadata
results/geometry.csv               one row per run, geometry metrics
results/analysis/correlations.csv  the correlation table
results/analysis/merged.csv        both tables joined, one row per run
results/meshes/<exp>/              exported meshes
results/logs/<exp>.log             full output of every stage
```

The two CSV files are versioned in git because they are small and both of us
need them. Meshes, logs, checkpoints and the DTU data are not.

---

## 10. Repository layout

```
scripts/
  00_check_env.py       environment self-check
  install.sh            build the conda environment
  dtu_prepare.py        DTU NeuS format to Nerfstudio transforms.json
  dtu_splits.py         generate the 75 train and test splits
  run_experiments.py    resumable batch runner (train, eval, export, stats)
  meshstat_helper.py    memory-isolated mesh statistics
  fetch_dtu_eval.py     pull only the needed DTU reference files
  eval_geometry.py      mesh to DTU millimetres, then the DTU metrics
  analyze.py            join the tables, rank correlation with bootstrap
  diag_frame.py         which coordinate frame is a mesh in
  diag_transform.py     wrong transform or junk in the mesh
  calibrate.sh          one full run, times it
  sweep.sh              iteration-count sweep
  01_check_reference.py validate a CAD STL as reference (Part B)
  02_test_metrics.py    validate the geometry metrics on synthetic data (Part B)
env/
  patch_torch_load.py   Nerfstudio checkpoint-load patch
  patch_printing.py     Nerfstudio stats-log patch
  patch_export.py       limit the Poisson export to the unit sphere
  patch_limit.py        add --exp and --limit to the runner
src/nerfpipe/metrics/
  dtu_eval.py           DTU protocol: accuracy, completeness, chamfer, F-score
  geometry.py           ICP plus Chamfer against a reference mesh (Part B)
data/dtu/
  neus/                 downloaded DTU scenes
  prepared/             converted scenes plus scale_mat.npy
  experiments/          the 75 split directories
  eval/                 official reference, Points/stl and ObsMask
results/                see section 9
docs/SETUP.md           longer setup notes
```

---

## 11. Working agreements

Code is shared through this repository, data through a USB stick. The DTU
reference does not need the stick, `fetch_dtu_eval.py` pulls it anywhere.

The full series of 75 runs is produced on one machine, see section 5.

Both result CSV files are committed. The runner writes the git commit hash
into every result row, which only means something if the working tree was
committed before the run.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied` on download | shell is in a /mnt/c path | `cd ~`, start WSL with `wsl ~` |
| `CondaToSNonInteractiveError` | channel terms not accepted | the `conda tos accept` lines in 3.3 |
| `sm_120 is not compatible` | PyTorch from cu118 not cu128 | reinstall PyTorch from the cu128 index |
| `Could not find compatible tinycudann extension for compute capability NN` | tiny-cuda-nn built for another GPU | rebuild with the auto-detected architecture, section 3.4 |
| `ld: cannot find -lcuda` on tcnn build | linker misses the WSL stub | the LDFLAGS line in 3.4 |
| `No module named pkg_resources` on tcnn build | setuptools too new | `pip install "setuptools<80"`, `--no-build-isolation` |
| `Weights only load failed` in ns-eval or ns-export | PyTorch 2.6 default | `python env/patch_torch_load.py` |
| training dies with `math domain error` | stats log glitch on WSL | `python env/patch_printing.py` |
| `nvidia-smi` empty or GPU access blocked | driver updated, needs restart | `wsl --shutdown`, then restart Windows if needed |
| training process `Killed` | WSL ran out of RAM | close other runs, mesh stats are memory-capped by design |
| mesh is huge, no vertices inside the unit sphere | Poisson export not limited | `python env/patch_export.py`, section 6 |
| mesh statistics columns stay empty | mesh too large, capped helper gave up | usually the same cause as the row above |
| `eval_geometry.py` reports `no_gt` | DTU reference missing | section 4.4 |
| geometry columns missing in `analyze.py` | old version, column name collision | pull the current `scripts/analyze.py` |
| `fetch_dtu_eval.py --dry-run` fails | server without range requests | download by hand, section 4.4 |
| COLMAP feature extraction fails (Part B) | no OpenGL in WSL | add `--no-gpu` |
