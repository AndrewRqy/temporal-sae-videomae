# Synthetic Ball Datasets

Synthetic video clips of a moving ball generated with [Kubric](https://github.com/google-research/kubric).
Each clip is 16 frames at 256×256 px with a known ground-truth speed profile, used to probe
which SAE features track temporal motion signals.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- GPU passthrough enabled (`--gpus all` requires NVIDIA Container Toolkit on Linux; on Windows, Docker Desktop with WSL2 backend)

Pull the Kubric image once before generating any dataset:

```powershell
docker pull kubricdockerhub/kubruntu
```

All scripts must be run from **this directory** (`data/`) in PowerShell.

---

## Datasets

| Dataset | Script | Clips | Speed profile |
|---|---|---|---|
| Velocity | `ball_dataset.py` | 392 | Constant (multiple speeds) |
| Acceleration | `ball_dataset.py` | 280 | Linearly increasing |
| Static | `static_ball_dataset.py` | 30 | 0 m/s (no motion) |
| Back-and-forth | `backforth_ball_dataset.py` | 50 | Reverses direction mid-clip |
| Slow-fast-slow | `nonmono_ball_dataset.py` | 50 | 1.5 → 6.0 → 1.5 m/s |
| Fast-slow-fast | `fastslow_ball_dataset.py` | 50 | 6.0 → 1.5 → 6.0 m/s |

---

## Running

### Velocity and acceleration (combined script)

```powershell
# Generate both velocity and acceleration datasets
.\run_ball_dataset.ps1

# Generate only one
.\run_ball_dataset.ps1 velocity
.\run_ball_dataset.ps1 acceleration
```

Output: `output/velocity/` and `output/acceleration/`

### Static

```powershell
.\run_static_dataset.ps1
```

Output: `output/static/`

### Back-and-forth

```powershell
.\run_backforth_dataset.ps1
```

Output: `output/backforth/`

### Slow-fast-slow (non-monotonic)

```powershell
.\run_nonmono_dataset.ps1
```

Output: `output/nonmono/`

### Fast-slow-fast

```powershell
.\run_fastslow_dataset.ps1
```

Output: `output/fastslow/`

---

## Output format

Each clip is saved as a folder named `dir{DD}_pos{PP}/` containing:

```
dir00_pos00/
├── rgba_00000.png   # frame 0
├── rgba_00001.png
├── ...
├── rgba_00015.png   # frame 15
└── metadata.json    # label, speed profile, trajectory
```

`metadata.json` includes the ground-truth speed at every frame and the full ball trajectory,
which the analysis scripts use to compute per-feature speed correlations.

---

## Transferring to the cluster

After generation, copy the output folders to the cluster for SAE analysis:

```powershell
scp -r "output\fastslow" renqy@fe.ds:~/sae-for-vlm/ball_dataset/
scp -r "output\nonmono"  renqy@fe.ds:~/sae-for-vlm/ball_dataset/
```

Then submit the corresponding job from `jobs/`:

```bash
sbatch jobs/analyze_fastslow.sh
sbatch jobs/analyze_nonmono.sh
```
