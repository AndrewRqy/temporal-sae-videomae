# Detecting Temporal Concepts in Video Transformers via Sparse Autoencoders

**Andrew Ren, Jianghuai Li** — TTIC 31280 Course Project

This repository extends the work of [Pach et al. (2025)](https://arxiv.org/abs/2504.02821) (*Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models*), adapting their SAE codebase to **VideoMAE** and introducing the **Neural Feature Probe (NFP) test** — a within-video covariance statistic that identifies SAE features encoding temporal concepts (speed, velocity, acceleration, direction) with a no-false-positives guarantee.

---

## Key results

| Condition | Significant features | Selectivity |
|---|---|---|
| DINOv2 SAE (negative control) | 0 / 6144 (0.00%) | — |
| VideoMAE SAE | 75 / 6144 (1.22%) | Diagonal dominant |
| Synthetic SAE (positive control) | 31 / 6144 (0.50%) | Diagonal dominant |

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11. The `dictionary_learning/` library is from [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning).

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
  nfp_test.py             NFP test for VideoMAE SAE
  nfp_test_dino_patch.py  NFP test for DINOv2 negative control
  nfp_test_synthetic.py   NFP test + alignment for synthetic positive control
  gen_synthetic_activations.py  Synthetic representation generator
  proj_fraction.py        Projection fraction metric (W_tau alignment)
data/                   NFP ball video dataset generation (Kubric/Docker)
training/               SAE training and activation extraction
  sae_train.py            Main SAE training entry point
  save_activations.py     VideoMAE activation extractor
  extract_dino_patch_activations.py  DINOv2 spatial patch extractor
eval/                   Monosemanticity metrics and evaluation
jobs/                   SLURM cluster job scripts (organized by experiment)
  dataset/                Dataset generation jobs
  videomae_sae/           VideoMAE SAE training + NFP test
  dino_negative_control/  DINOv2 negative control pipeline
  synthetic_100/          Synthetic positive control (100 static dirs)
  synthetic_763/          Synthetic robustness variant (763 static dirs)
  monosemanticity/        Monosemanticity score evaluation
results/                Experiment output files (.txt; .pkl files gitignored)
dictionary_learning/    SAE library (third-party, from saprmarks)
models/                 Model wrappers including VideoMAE
```

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for a full map of every experiment to its scripts, inputs, outputs, and results.

---

## Running experiments

All jobs are SLURM batch scripts. Submit with `sbatch <script>` from `~/sae-for-vlm/` on the cluster. Scripts reference Python entry points at the repo root (`sae_train.py`, `save_activations.py`, `extract_dino_patch_activations.py`).

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
