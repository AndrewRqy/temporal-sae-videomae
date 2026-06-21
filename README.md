# Detecting Temporal Concepts in Video Transformers via Sparse Autoencoders

**Andrew Ren, Jianghuai Li** — TTIC 31280 Course Project

This repository extends the work of [Pach et al. (2025)](https://arxiv.org/abs/2504.02821) (*Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models*), adapting their SAE codebase to **VideoMAE** and introducing the **Neural Feature Probe (NFP) test** — a within-video covariance statistic that identifies SAE features encoding temporal concepts (speed, velocity, acceleration, direction) with a no-false-positives guarantee.

---

## Key results

| Condition | Tau mode | Significant features | Selectivity |
|---|---|---|---|
| DINOv2 SAE (negative control) | first_frame | 0 / 6144 (0.00%) | — |
| VideoMAE SAE | first_frame | 75 / 6144 (1.22%) | Diagonal dominant |
| VideoMAE SAE | avg_frames | 151 / 6144 (2.46%) | Diagonal dominant |
| Synthetic SAE (positive control) | first_frame | 31 / 6144 (0.50%) | Diagonal dominant |

The `first_frame` mode assigns each tubelet's tau value from the first of its two
constituent frames; `avg_frames` averages across both frames. First-frame results
are the primary comparison across conditions. Avg-frames is reported as a
methodological variant for VideoMAE only (see §4.2 of the paper).

### Linear-decomposition baselines (PCA / ICA / raw)

To isolate what the SAE specifically contributes, we run the **same** monosemanticity (MS)
and NFP tests on **PCA**, **ICA**, and the **raw** layer-11 dimensions (PCA/ICA wrapped in the
SAE `Dictionary` interface, so every stage is reused unchanged). See `EXPERIMENTS.md` §5b–§5d.

| Filter | MS score (peak) | NFP significant | Notes |
|---|---|---|---|
| **SAE** (6144) | **0.475 ± 0.063** (0.80) | **75 / 6144 (1.2%)** | high-MS tail; sparse, selective |
| Raw layer (768) | 0.469 ± 0.007 (0.49) | 698 / 768 (90.9%) | dense, not selective |
| PCA (512, sign-split) | 0.467 ± 0.006 (0.50) | 351 / 512 (68.6%) | dense, selective |
| ICA (512, sign-split) | 0.467 ± 0.009 (0.51) | 352 / 512 (68.8%) | dense, selective |

**Takeaway:** PCA/ICA/raw match the SAE's *mean* MS but lack its high-MS tail (peak ~0.50 vs 0.80),
and they flag 69–91% of features as temporal vs the SAE's 1.2%. A dimensionality sweep
(`analysis/sweep_pca_ica_dim.py`) shows neither metric moves toward the SAE at any D. The SAE's
advantage — a few highly-monosemantic, sparsely-localized temporal features — is intrinsic to
sparse overcomplete coding, not achievable by any linear decomposition. (Numbers above are from a
local single-GPU run; component count / fit corpus differ slightly from a full-rank cluster run.)

---

## Setup

```bash
pip install -r requirements.txt   # use requirements-win.txt on Windows
pip install av                    # PyAV — decodes SSv2/ball .webm/.png (see note below)
```

Python 3.11. The `dictionary_learning/` library is from [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning),
extended here with `PCADict` / `ICADict` / `IdentityDict` so PCA, ICA, and the raw layer can be
swapped in wherever an SAE is used. `nnsight` is only needed for SAE *training* (the package
import is guarded, so extraction / PCA-ICA / NFP run without it).

For the NFP ball video dataset, render locally with Kubric via Docker:

```powershell
cd data
.\run_nfp_dataset.ps1    # renders 3000 synthetic ball videos to ./output/nfp/
```

Then SCP the output to the cluster before running any cluster jobs that require the NFP dataset.

---

## Repository structure

```
analysis/               NFP test and analysis scripts
  nfp_test.py             NFP test (--sae_model standard|pca|ica|identity)
  nfp_test_dino_patch.py  NFP test for DINOv2 negative control
  nfp_test_synthetic.py   NFP test + alignment for synthetic positive control
  fit_pca_ica.py          Fit PCA + ICA decompositions on layer-11 activations
  sweep_pca_ica_dim.py    Sweep #components D; report MS + NFP vs D
  gen_synthetic_activations.py  Synthetic representation generator
  proj_fraction.py        Projection fraction metric (W_tau alignment)
data/                   NFP ball video dataset generation (Kubric/Docker)
training/               SAE training and activation extraction
  train_sae.py            Main SAE training entry point
  extract_activations.py  VideoMAE activation extractor (--sae_model standard|pca|ica|…);
                          deployed flat to the cluster as save_activations.py
  extract_dino_patch_activations.py  DINOv2 spatial patch extractor
eval/                   Monosemanticity metric (metric.py) and evaluation
jobs/                   SLURM cluster job scripts (organized by experiment)
  dataset/                Dataset generation jobs
  videomae_sae/           VideoMAE SAE training + NFP test
  dino_negative_control/  DINOv2 negative control pipeline
  synthetic_100/          Synthetic positive control (100 static dirs)
  synthetic_763/          Synthetic robustness variant (763 static dirs)
  monosemanticity/        MS score: SAE, raw, and PCA/ICA baselines + fit_pca_ica.sh
  local/                  Single-GPU (no-SLURM) runners for the PCA/ICA MS + NFP baselines
results/                Experiment output files (.txt; .pkl files gitignored)
local_runs/             Local single-GPU run outputs (gitignored)
dictionary_learning/    SAE library (third-party); adds PCADict/ICADict/IdentityDict
models/                 Model wrappers including VideoMAE
```

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for a full map of every experiment to its scripts, inputs, outputs, and results.

---

## Running experiments

**On the cluster:** all jobs are SLURM batch scripts. Submit with `sbatch <script>` from
`~/sae-for-vlm/`. Scripts are SCP'd flat to the repo root, where the Python entry points
`train_sae.py`, `save_activations.py` (the flattened copy of `training/extract_activations.py`),
and `extract_dino_patch_activations.py` are invoked.

**Locally (single GPU, no SLURM):** the PCA/ICA monosemanticity and NFP baselines run end-to-end
on one machine with the full SSv2 + NFP datasets present:

```powershell
# 1. MS for PCA/ICA: fits pca.pt/ica.pt, caches train acts + DINOv2 embeddings, scores MS
powershell -ExecutionPolicy Bypass -File jobs\local\run_pca_ica_ms_local.ps1
# 2. NFP for raw/PCA/ICA (reuses the fits from step 1)
powershell -ExecutionPolicy Bypass -File jobs\local\run_nfp_local.ps1
# 3. Dimensionality sweep (MS + NFP vs number of components)
$env:PYTHONPATH = (Get-Location).Path
python analysis/sweep_pca_ica_dim.py --ssv2_path <SSv2> --nfp_dir data/output/nfp `
    --train_dir local_runs/train_acts --embeds_path local_runs/embeds/ssv2_val_dinov2.pt `
    --output_csv local_runs/sweep_dim.csv --grid 16 32 64 128 256 512
```

Set the `-Py` path inside the runners to your `torch`+CUDA interpreter. Outputs land under
`local_runs/`. See [`EXPERIMENTS.md`](EXPERIMENTS.md) (§5b–§5d and the "Local execution" section)
for the complete execution order, prerequisites, and results.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the complete execution order and prerequisites.

---

## Original paper

This project builds on:

```bibtex
@article{pach2025sparse,
  title={Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models},
  author={Mateusz Pach and Shyamgopal Karthik and Quentin Bouniot and Serge Belongie and Zeynep Akata},
  journal={arXiv preprint arXiv:2504.02821},
  year={2025}
}
```
