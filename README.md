# Temporal Feature Localization in Video Models via Sparse Autoencoders

**Andrew Ren, Jianghuai Li** — TTIC 31280 Course Project

This repository extends the work of [Pach et al. (2025)](https://arxiv.org/abs/2504.02821) (*Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models*), which applied SAEs to CLIP and other vision-language models. We adapt their codebase to **VideoMAE**, adding support for video transformer activations and a suite of synthetic motion datasets to probe which SAE features encode temporal concepts.

Our additions on top of the original repo:
- **VideoMAE compatibility** — model wrapper and activation hooks for `MCG-NJU/videomae-base-finetuned-ssv2`
- **Synthetic ball datasets** (`data/`) — six Kubric-generated motion datasets with controlled speed profiles
- **Temporal analysis scripts** (`analysis/`) — per-feature speed correlation, reversal sensitivity, and ablation analysis
- **Cluster job scripts** (`jobs/`) — SLURM scripts for training and analysis on the TTIC compute cluster

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11. The `dictionary_learning/` library is from [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning).

---

## Repository structure

```
data/               Kubric synthetic dataset generation scripts
analysis/           SAE feature analysis on video datasets
training/           SAE training and activation extraction
eval/               Monosemanticity metrics and evaluation
visualization/      Plotting and result visualization
steering/           Feature steering experiments (inherited from original repo)
utils/              Shared helpers
jobs/               SLURM cluster job scripts
results/            Output .pkl files (gitignored)
dictionary_learning/ SAE library (third-party)
models/             Model wrappers including VideoMAE
```

---

## Synthetic datasets

See [`data/README.md`](data/README.md) for full instructions. Datasets are generated with Kubric via Docker:

```powershell
cd data
.\run_nonmono_dataset.ps1    # slow-fast-slow, 50 clips
.\run_fastslow_dataset.ps1   # fast-slow-fast, 50 clips
```

---

## Cluster jobs (`jobs/`)

All scripts are SLURM batch jobs. Submit with `sbatch <script>` from the repo root on the cluster.

### Training

| Script | What it does |
|---|---|
| `train_sae.sh` | Trains a **BatchTopK SAE** on VideoMAE layer-11 activations extracted from SSv2. Runs multiple expansion factors and layer configurations. |
| `train_sae_standard.sh` | Same as above but trains a **standard (non-TopK) SAE** for comparison against the BatchTopK variant. |

### Analysis

| Script | What it does |
|---|---|
| `analyze_ball.sh` | Runs `analysis/analyze_ball.py` — general feature extraction on all ball dataset clips, recording per-video SAE activations without temporal aggregation. |
| `analyze_vel_accel.sh` | Runs `analysis/analyze_vel_accel.py` — temporal speed-correlation analysis on the **velocity** and **acceleration** datasets (constant and linearly-increasing speed profiles). Computes per-feature Pearson correlation with the known ground-truth speed trajectory. |
| `analyze_decel.sh` | Same pipeline as `analyze_vel_accel.sh` but applied to a **deceleration** (linearly-decreasing) dataset, testing whether features that track acceleration also track its reverse. |
| `analyze_nonmono.sh` | Runs `analysis/analyze_nonmono.py` — temporal analysis on the **slow-fast-slow** dataset. The reference speed profile is `[1.5, 1.5, 3.75, 6.0, 6.0, 3.75, 1.5, 1.5]` m/s at the tubelet level. Outputs `results/nonmono_{btk,std}.pkl`. |
| `analyze_fastslow.sh` | Runs `analysis/analyze_fastslow.py` — same pipeline on the **fast-slow-fast** dataset (mirror of nonmono). Used to check whether top nonmono features flip sign, confirming they track the speed trajectory rather than just firing during fast motion. Outputs `results/fastslow_{btk,std}.pkl`. |
| `analyze_ablation.sh` | Runs `analysis/analyze_ablation.py` — analysis on the **static** and **back-and-forth** control datasets. Static clips (no motion) serve as a negative control; back-and-forth tests direction-reversal sensitivity. |

### Evaluation

| Script | What it does |
|---|---|
| `eval_mono.sh` | Computes the **Monosemanticity Score** for VideoMAE SAEs using the DINOv2-based metric from the original paper. Evaluates multiple SAE configurations across layers. |
| `matryoshka_hierarchy.sh` | Analyzes the hierarchical structure that emerges in **Matryoshka BatchTopK SAEs** — checks whether the high-level feature group concentrates semantically coarser concepts than the low-level group. |
| `monosemanticity_score.sh` | Extended monosemanticity scoring across a sweep of models, layers, and expansion factors (inherited from the original repo). |
| `mllm_steering.sh` | Steering experiments using SAE features to intervene on LLaVA outputs via the vision encoder (inherited from the original repo, not the focus of this project). |

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
