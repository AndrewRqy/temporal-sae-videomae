# Experiment Registry

Maps each experiment in the paper to its scripts. Scripts are organized under
`jobs/<experiment>/`. On the cluster all scripts live flat at `~/sae-for-vlm/`
(SLURM requires flat paths); SCP each script there before submitting.

---

## 0. Dataset Generation (`jobs/dataset/`)

### 0a. NFP Ball Video Dataset (local, Docker/Kubric)

**What it is:** 3,000 synthetic ball videos (16 frames, 224×224) with ground-truth
tau metadata (speed, vel_x, vel_y, accel_mag, direction) and spatial token indices.
Rendered via Kubric/Blender inside a Docker container.

**Scripts:**
- `data/nfp_ball_dataset.py` — Python generator (run inside Kubric container)
- `data/run_nfp_dataset.ps1` — PowerShell launcher (runs Docker, outputs to `./output/nfp/`)

**Output:** `/net/scratch2/renqy/nfp/v00000/ … v02999/` (transferred to cluster after local generation)

**Note:** Run locally before any cluster jobs that require the NFP dataset.

---

### 0b. VideoMAE Layer-11 Activations from SSv2 (`jobs/dataset/run_extract_videomae_activations.sh`)

**What it is:** Extracts all 1568 token representations per video (no pooling)
from VideoMAE layer 11 (post-MLP residual) on SSv2. Used as training data for
the VideoMAE SAE.

**Script:** `jobs/dataset/run_extract_videomae_activations.sh`

| Input | Output |
|-------|--------|
| SSv2 at `/net/scratch/renqy/SSv2` | `/net/scratch/renqy/activations/{train,val}` |

---

### 0c. Synthetic Representations — 100 Static (`jobs/dataset/run_gen_synthetic.sh`)

**What it is:** Generates synthetic 768-dim representations from NFP ball video
metadata. Each h(V,t) = tau_norm @ W_tau + static_code @ W_sigma + noise.
Used as training/val data for the synthetic positive control SAE.

**Script:** `jobs/dataset/run_gen_synthetic.sh`

| Input | Output |
|-------|--------|
| NFP metadata at `/net/scratch2/renqy/nfp` | `/net/scratch2/renqy/synthetic_acts/{train,val,all_videos.pt,matrices.pt}` |

---

### 0d. Synthetic Representations — 763 Static (`jobs/dataset/run_gen_synthetic_763.sh`)

**What it is:** Same as 0c but with n_static=763 so W_tau + W_sigma span all of R^768.
Used for the robustness variant of the synthetic experiment.

**Script:** `jobs/dataset/run_gen_synthetic_763.sh`

| Input | Output |
|-------|--------|
| NFP metadata at `/net/scratch2/renqy/nfp` | `/net/scratch2/renqy/synthetic_acts_763/{train,val,all_videos.pt,matrices.pt}` |

---

### 0e. SSv2 DINOv2 Embeddings for MS Scoring (`jobs/dataset/run_ssv2_dino_embeddings.sh`)

**What it is:** Computes max-pooled DINOv2-base embeddings for 800 SSv2 val clips.
Used as the vision encoder for monosemanticity scoring (not needed for NFP tests).

**Script:** `jobs/dataset/run_ssv2_dino_embeddings.sh`

| Input | Output |
|-------|--------|
| SSv2 val at `/net/scratch/renqy/SSv2` | `/net/scratch/renqy/embeddings/ssv2_val_dinov2.pt` |

---

## 1. VideoMAE SAE (`jobs/videomae_sae/`)

**What it tests:** Trains a standard SAE on VideoMAE layer-11 representations from
SSv2, then runs the NFP test to find features whose within-video covariance with
tau variables is significantly nonzero. Primary result of the paper.

**Paper sections:** §3.2, §3.3, §4.2

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Train SAE | `jobs/videomae_sae/run_sae_train_standard_deadpen.sh` | `/net/scratch/renqy/activations/{train,val}` | `/net/scratch/renqy/sae_checkpoints_deadpen_0p03/` |
| 2. NFP test | `jobs/videomae_sae/run_nfp_test.sh` | NFP dataset + SAE above | `/net/scratch2/renqy/nfp_results/videomae_deadpen_0p03.pt` |

**Results:** 75/6144 significant (1.22%), diagonal-dominant selectivity matrix.

**Notes:**
- SAE: expansion ×8 → 6144 features, l1=0.1, dead_penalty=0.03, 15,320 steps, batch 4096.
- Prerequisite: run `jobs/dataset/run_extract_videomae_activations.sh` first.

---

## 2. DINOv2 Negative Control (`jobs/dino_negative_control/`)

**What it tests:** Trains an SAE on DINOv2 spatial patch tokens at the
ball-containing grid position — same 14×14 grid and ball-tracking rule as
VideoMAE. DINOv2 has no temporal context, so the NFP test must find 0 significant
features.

**Paper sections:** §3.2, §3.3, §4.1

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Extract patch tokens | `jobs/dino_negative_control/run_save_dino_patch_activations.sh` | SSv2 at `/net/scratch/renqy/SSv2` | `/net/scratch2/renqy/dino_patch_activations/{train,val}` |
| 2. Train SAE | `jobs/dino_negative_control/run_sae_train_dino_patch.sh` | Activations above | `/net/scratch2/renqy/sae_checkpoints_dino_patch/` |
| 3. NFP test | `jobs/dino_negative_control/run_nfp_test_dino_patch.sh` | NFP dataset + SAE above | `/net/scratch2/renqy/nfp_results/dino_patch_nfp.pt` |

**Results:** 0/6144 significant (0.00%), mean |C_i| = 7.0×10⁻⁵.

**Notes:**
- Train activations: ~2.5M tokens (50 chunks of 50k) after disk-quota trimming.
  Val dir contains 1 chunk (copied from train) — sufficient for diagnostic monitoring.
- 15,320 steps matches VideoMAE SAE for fair comparison.
- **Supersedes** the old CLS/pooler-output pipeline:
  `run_save_dino_activations.sh` + `run_sae_train_dino.sh` + `run_nfp_test_dino.sh`.
  Those scripts used DINOv2's pooler output (CLS-derived, spatially aggregated),
  which introduces a confound vs VideoMAE's spatial ball token.

---

## 3. Synthetic Positive Control — 100 Static (`jobs/synthetic_100/`)

**What it tests:** Generates synthetic representations with known structure
(5 temporal W_tau + 100 static W_sigma + noise), trains an SAE, and runs the NFP
test. Validates that the test correctly detects temporal encoding when it is present,
and that significant features align with W_tau. Also computes the projection
fraction metric.

**Paper sections:** §3.5, §4.3

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Generate data | `jobs/synthetic_100/run_gen_synthetic.sh` | NFP metadata at `/net/scratch2/renqy/nfp` | `/net/scratch2/renqy/synthetic_acts/` |
| 2. Train SAE | `jobs/synthetic_100/run_sae_train_synthetic.sh` | Activations above | `/net/scratch2/renqy/sae_checkpoints_synthetic/` |
| 3. NFP test | `jobs/synthetic_100/run_nfp_test_synthetic.sh` | `all_videos.pt`, `matrices.pt`, SAE above | `/net/scratch2/renqy/nfp_results/synthetic_nfp.pt` |
| 4. Proj fraction | `jobs/synthetic_100/run_proj_fraction.sh` | SAE + `matrices.pt` + NFP results | stdout log |

**Results:** 31/6144 significant (0.50%), diagonal-dominant selectivity.
Proj-frac W_tau: 22.9% sig vs 0.4% non-sig (60× ratio).

**Notes:**
- Seed 42, n_static=100, noise_scale=0.05, 80/20 train/val split.
- Prerequisite: NFP dataset must exist at `/net/scratch2/renqy/nfp`.

---

## 4. Synthetic Robustness — 763 Static (`jobs/synthetic_763/`)

**What it tests:** Repeats Experiment 3 with n_static=763 so W_tau + W_sigma span
all of R^768. Tests whether the NFP covariance statistic is robust when static
content dominates 99.3% of representation space.

**Paper sections:** §4.3 (robustness paragraph)

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Generate data | `jobs/synthetic_763/run_gen_synthetic_763.sh` | NFP metadata at `/net/scratch2/renqy/nfp` | `/net/scratch2/renqy/synthetic_acts_763/` |
| 2. Train SAE | `jobs/synthetic_763/run_sae_train_synthetic_763.sh` | Activations above | `/net/scratch2/renqy/sae_checkpoints_synthetic_763/` |
| 3. NFP test | `jobs/synthetic_763/run_nfp_test_synthetic_763.sh` | `all_videos.pt`, `matrices.pt`, SAE above | `/net/scratch2/renqy/nfp_results/synthetic_763_nfp.pt` |

**Results:** 31/6144 significant (0.50%) — identical to 100-static, confirming
robustness. Proj-frac collapses to 1.4× (geometric artifact, not a test failure).

---

## 5. Monosemanticity Score (`jobs/monosemanticity/`)

**What it tests:** Computes MS score for VideoMAE SAE features and for raw
(pre-SAE) layer-11 activations as a baseline. MS measures interpretability via
weighted pairwise DINOv2-embedding cosine similarity of maximally activating clips.

**Paper sections:** §3.3

| Script | Input | Output |
|--------|-------|--------|
| `jobs/monosemanticity/eval_mono_auxk.sh` | SAE checkpoint, DINOv2 embeddings | MS score under `sae_activations/standard_deadpen_0p03_val/ms_dinov2/` |
| `jobs/monosemanticity/eval_mono_raw.sh` | DINOv2 embeddings | MS score under `sae_activations/raw_val/ms_dinov2/` |

**Results:** SAE: 0.475 ± 0.063. Raw baseline: TBD (job pending).

**Notes:**
- Both scripts call `eval/metric.py` (not `metric.py` — correct path relative to project root).
- Prerequisites: DINOv2 embeddings must exist (run `jobs/dataset/run_ssv2_dino_embeddings.sh`).
- `eval_mono_auxk.sh` also requires VideoMAE SAE checkpoint.
