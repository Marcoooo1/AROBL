# Arbeitsumgebung einrichten

Projekt: Objektrekonstruktion mit Neural Radiance Fields.
Gruppe 7, Applied Robot Learning, FH Technikum Wien.
Marco Bayreder, Lorenz Christian.

Diese Anleitung richtet die Umgebung auf einem Windows Rechner mit NVIDIA GeForce
RTX 5070 unter WSL2 ein. Rechne mit 60 bis 90 Minuten, der Großteil davon sind Downloads.

## Warum nicht die offizielle Nerfstudio Anleitung

Die RTX 5070 ist eine Blackwell Karte mit Compute Capability `sm_120`. Blackwell wird
erst ab CUDA 12.8 unterstützt. Die offizielle Nerfstudio Anleitung nennt CUDA 11.8 und
PyTorch 2.1.2 und führt auf dieser Karte zwangsläufig in eine Fehlerkette: erst die
Meldung dass `sm_120` nicht zur PyTorch Installation passt, dann ein `tiny-cuda-nn` Build
der an der CUDA Versionsabweichung scheitert, dann GCC das für CUDA 11.8 zu neu ist.

Daraus folgen zwei Abweichungen:

1. PyTorch kommt vom **cu128** Wheel Index, nicht von cu118.
2. `tiny-cuda-nn` wird **nicht** gebaut. Nerfacto bringt Hash Encoding und MLP auch als
   reine PyTorch Variante mit, wählbar über `implementation="torch"`. Das kostet
   Trainingszeit, nimmt aber die häufigste Fehlerquelle aus dem Weg.

## Warum WSL2 und nicht Dual Boot

Bei WSL2 liefert der Windows Treiber die GPU durch, in Ubuntu wird nur das CUDA Toolkit
installiert. Beim Dual Boot müssten Secure Boot, Partitionierung und bei einem Laptop
zusätzlich die Hybridgrafik selbst in Ordnung gebracht werden, ohne messbaren Vorteil
für dieses Projekt.

---

## Schritt 1: Windows vorbereiten

PowerShell als Administrator.

Aktuellen GeForce Treiber installieren, dann prüfen:

```powershell
nvidia-smi
```

Die Ausgabe muss die RTX 5070 zeigen und oben rechts CUDA 12.8 oder höher nennen. Diese
Zahl ist die vom Treiber maximal unterstützte Version, nicht die installierte.

```powershell
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
```

Nach dem Neustart Benutzername und Passwort anlegen.

**Zwei Dinge nie tun:** keinen NVIDIA Treiber innerhalb von WSL installieren, das
überschreibt die durchgereichten Bibliotheken. Und das Projekt nicht unter `/mnt/c/`
ablegen, Dateizugriffe über die Windowsgrenze sind langsam.

## Schritt 2: Ubuntu Basis

WSL immer mit `wsl ~` starten, sonst landet man im Windows Systemverzeichnis und darf
dort nicht schreiben.

```bash
wsl ~
nvidia-smi
```

Wenn hier die GPU erscheint, funktioniert der Durchgriff. Wenn nicht, ist der Windows
Treiber zu alt oder WSL läuft noch in Version 1, prüfbar mit `wsl -l -v`.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl unzip ffmpeg colmap
```

## Schritt 3: CUDA Toolkit 12.8

Nicht `apt install nvidia-cuda-toolkit` verwenden. Das Ubuntu Paket installiert eine
ältere CUDA Version, deren `nvcc` die Architektur `sm_120` nicht kennt.

```bash
cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

Pfade dauerhaft setzen:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

`nvcc --version` muss `release 12.8` melden.

## Schritt 4: Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

Danach steht `(base)` vor dem Prompt.

conda verlangt beim ersten Anlegen einer Umgebung eine Zustimmung zu den Bedingungen der
Anaconda Standardkanäle. Studium und Lehre sind von der Lizenzpflicht ausgenommen:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda config --add channels conda-forge
conda config --set channel_priority strict
```

`defaults` bleibt dabei in der Kanalliste stehen und lässt sich nicht entfernen, das ist
bei conda so eingebaut und unkritisch. Mit `strict` und conda-forge an erster Stelle
kommen die Pakete trotzdem von conda-forge.

## Schritt 5: Projektordner

```bash
mkdir -p ~/projects/nerf-object-reconstruction && cd $_
git init

mkdir -p .vscode configs/experiments docs scripts tests notebooks
mkdir -p data/{raw,masked,colmap,reference} outputs results
mkdir -p src/nerfpipe/metrics
touch src/nerfpipe/__init__.py src/nerfpipe/metrics/__init__.py
touch data/{raw,masked,colmap,reference}/.gitkeep outputs/.gitkeep results/.gitkeep
```

Bedeutung der Ordner:

| Ordner | Inhalt |
|--------|--------|
| `data/raw/` | Master Datensatz pro Objekt, unverändert |
| `data/masked/` | freigestellte Bilder nach SAM 2 |
| `data/colmap/` | COLMAP Workspaces und `transforms.json` |
| `data/reference/` | Druck STL der Testobjekte als Ground Truth |
| `outputs/` | Nerfstudio Trainingsläufe und Checkpoints |
| `results/` | Meshes, STL Exporte, Metriktabellen |
| `src/nerfpipe/` | Pipeline Module |
| `scripts/` | ausführbare Einstiegspunkte |

`data/` und `outputs/` sind von Git ausgenommen, nur `data/reference/` wird versioniert.

## Schritt 6: Dateien anlegen

Die Dateien `pyproject.toml`, `scripts/install.sh`, `scripts/00_check_env.py`,
`.gitignore` und die drei Dateien unter `.vscode/` anlegen.

## Schritt 7: Installieren

```bash
cd ~/projects/nerf-object-reconstruction
bash scripts/install.sh 2>&1 | tee /tmp/install.log
```

Dauert zehn bis zwanzig Minuten, PyTorch mit CUDA 12.8 sind mehrere Gigabyte. Das Skript
bricht ab, wenn die PyTorch Installation keine `sm_120` Kernel enthält. Genau das ist
der Punkt, an dem es auf dieser Karte sonst still schiefgeht.

Bei einem Abbruch steht die Ursache im Protokoll:

```bash
tail -40 /tmp/install.log
```

## Schritt 8: Prüfen

```bash
conda activate nerf-recon
python scripts/00_check_env.py
```

Erwartung: alles grün außer `tiny-cuda-nn`, das soll als Warnung stehen bleiben.

---

## Täglicher Start

```bash
wsl ~
cd ~/projects/nerf-object-reconstruction
conda activate nerf-recon
```

Der Prompt muss `(nerf-recon)` zeigen. Steht dort `(base)`, ist die Umgebung nicht aktiv
und nichts funktioniert. Das ist der häufigste Stolperstein im Alltag.

Kürzel anlegen:

```bash
echo "alias nerf='cd ~/projects/nerf-object-reconstruction && conda activate nerf-recon'" >> ~/.bashrc
source ~/.bashrc
```

Danach genügt `wsl ~` und dann `nerf`.

Nichts geht verloren wenn das Terminal geschlossen wird. WSL2 ist eine dauerhafte
Installation, Umgebung, Pakete und Dateien bleiben in Ubuntu.

## VS Code

VS Code läuft unter Windows, Code und Interpreter liegen in WSL2.

1. VS Code unter **Windows** installieren, nicht in WSL.
2. Erweiterung **WSL** (`ms-vscode-remote.remote-wsl`) installieren.
3. Im Projektordner `code .` aufrufen. Beim ersten Mal installiert VS Code seinen Server
   in WSL.
4. Unten links muss `WSL: Ubuntu-24.04` stehen. Steht dort nichts, arbeitet man in
   Windows und findet den Interpreter nicht.
5. `Strg+Shift+P`, dann `Python: Select Interpreter`, dann
   `~/miniconda3/envs/nerf-recon/bin/python`.

Später genügt `Datei` und `Zuletzt verwendet`, dort steht das Projekt mit dem Vermerk
`WSL`. VS Code startet WSL selbst, eine PowerShell braucht man dann nicht.

## Erster Testlauf

Mit einem fremden Datensatz prüfen, bevor eigene Bilder aufgenommen werden:

```bash
ns-download-data nerfstudio --capture-name=poster
ns-train nerfacto --data data/nerfstudio/poster \
  --pipeline.model.implementation torch \
  --output-dir outputs
```

Der Viewer läuft auf Port 7007. WSL2 leitet localhost nach Windows weiter, also im
Windows Browser `http://localhost:7007` öffnen. Falls das nicht greift, die WSL IP mit
`hostname -I` ermitteln und diese statt localhost verwenden.

## tiny-cuda-nn nachrüsten

Optional, erst wenn die Pipeline vollständig durchläuft:

```bash
conda activate nerf-recon
pip install ninja
export TCNN_CUDA_ARCHITECTURES=120
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

`TCNN_CUDA_ARCHITECTURES=120` ist der entscheidende Teil, ohne den der Build die
Zielarchitektur falsch ableitet. Scheitert es, bleibt der Torch Pfad funktionsfähig.
Der Zeitvergleich zwischen beiden Varianten ist eine brauchbare Randnotiz für die
Methodik.

---

## Häufige Fehler

| Meldung | Ursache | Lösung |
|---------|---------|---------|
| `Permission denied` beim Download | Prompt steht in `/mnt/c/...` | `cd ~`, WSL mit `wsl ~` starten |
| `CondaToSNonInteractiveError` | Bedingungen nicht akzeptiert | `conda tos accept` wie in Schritt 4 |
| `EnvironmentNameNotFound: nerf-recon` | `install.sh` nie gelaufen oder abgebrochen | Skript starten, `/tmp/install.log` prüfen |
| `sm_120 is not compatible with the current PyTorch installation` | PyTorch mit cu118 statt cu128 | PyTorch entfernen, mit cu128 Index neu installieren |
| `The detected CUDA version mismatches the version used to compile PyTorch` | System CUDA und PyTorch CUDA weichen ab | beides auf 12.8 bringen |
| `nvcc: Value 'sm_120' is not defined` | CUDA Toolkit zu alt | `apt` Variante entfernen, Toolkit 12.8 installieren |
| COLMAP bricht bei Feature Extraction ab | kein OpenGL Kontext in WSL | `--SiftExtraction.use_gpu 0` setzen, bei 5 bis 50 Bildern unkritisch |
| `nvidia-smi` leer in WSL | Treiber in WSL installiert oder WSL1 | Treiber nur unter Windows, `wsl --set-version Ubuntu-24.04 2` |
| Import oder `ns-train` nicht gefunden | Umgebung nicht aktiv | `conda activate nerf-recon` |
| Training sehr langsam | Daten liegen unter `/mnt/c/` | Projekt nach `~/projects/` verschieben |
