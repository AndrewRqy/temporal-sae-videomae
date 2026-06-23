# Experiment Registry

Maps each experiment in the paper to its scripts. Scripts are organized under
`jobs/<experiment>/`. On the cluster all scripts live flat at `~/sae-for-vlm/`
(SLURM requires flat paths); SCP each script there before submitting.

---

## Local execution (no cluster, single GPU)

The linear-decomposition baselines (§5b MS, §5c NFP, §5d sweep) can run end-to-end on a
single machine with a CUDA GPU and the full SSv2 + NFP datasets present locally — no SLURM.
PowerShell runners live in `jobs/local/`:

| Runner | Produces | Covers |
|--------|----------|--------|
| `jobs/local/run_pca_ica_ms_local.ps1` | `local_runs/decomp/{pca,ica}.pt`, MS scores | §5b (MS for PCA/ICA) |
| `jobs/local/run_nfp_local.ps1` | `local_runs/nfp_results/*.pt` | §5c (NFP for raw/PCA/ICA) |
| `analysis/sweep_pca_ica_dim.py` | `local_runs/sweep_dim.csv` | §5d (D sweep) |

**Local setup notes (one-time):**
- Use the interpreter that has `torch`+CUDA / `sklearn` / `transformers` (e.g. a conda env).
  The runners pin `-Py "C:\…\python.exe"` — set it to your interpreter.
- `pip install av` (PyAV; required to decode SSv2/ball `.webm`/`.png` — see requirements).
- Repo root must be on `PYTHONPATH` when invoking subdir scripts directly
  (`$env:PYTHONPATH = (Get-Location).Path`); the runners set this for you.
- `nnsight` is **not** required for raw/PCA/ICA (the package import is guarded); it is only
  needed for SAE *training*.
- `models/videomae.py`'s layer wrapper accepts `*args`, so it tolerates `transformers`
  versions that pass `head_mask` positionally (newer) or by keyword (the cluster's pin).
- The NFP ball dataset (`data/output/nfp/`, ~710 MB) is rendered locally via
  `data/run_nfp_dataset.ps1` and is gitignored — regenerate it if absent.
- All local outputs go under `local_runs/` (gitignored).

Order: run `run_pca_ica_ms_local.ps1` first (it fits `pca.pt`/`ica.pt` and caches
`train_acts` + DINOv2 embeddings that the NFP run and the sweep reuse).

**Skip the heavy steps via HuggingFace.** The trained **SAEs** (`weights/sae/`), fitted
`pca.pt`/`ica.pt`, the NFP dataset, synthetic ground truth (`synthetic/`), cached `train_acts`,
DINOv2 embeddings, and the NFP result tensors are hosted (public) at
[`AndrewRqy/temporal-sae-videomae`](https://huggingface.co/datasets/AndrewRqy/temporal-sae-videomae).
Download them into the matching local paths (`data/output/nfp/`, `local_runs/{sae,decomp,train_acts,embeds,synth_data,...}/`)
to reproduce §1–§5 without re-training, re-rendering, or re-running VideoMAE. The upload script is
`local_runs/hf_upload/upload_to_hf.py` and the full file map is in the dataset card.

> The `cluster_nfp_results/` tensors are verification-only — every experiment is reproducible from
> the hosted weights + data; they let you diff against the published numbers without recomputing.

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

### 1a. Train SAE + NFP test (first-frame tau, primary result)

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Train SAE | `jobs/videomae_sae/run_sae_train_standard_deadpen.sh` | `/net/scratch/renqy/activations/{train,val}` | `/net/scratch/renqy/sae_checkpoints_deadpen_0p03/` |
| 2. NFP test | `jobs/videomae_sae/run_nfp_test.sh` | NFP dataset + SAE above | `/net/scratch2/renqy/nfp_results/videomae_deadpen_0p03.pt` |

**Results:** 75/6144 significant (1.22%), diagonal-dominant selectivity matrix.

### 1b. NFP test — avg-frames tau (methodological variant)

Each VideoMAE tubelet spans 2 consecutive frames (16 frames → 8 temporal tokens).
The default `first_frame` mode assigns each tubelet the tau values of its first frame.
The `avg_frames` mode instead averages tau across both frames of the tubelet, giving
a smoother signal that better represents the tubelet's temporal midpoint.

**This uses the same trained SAE from step 1a — no retraining needed.**

| Step | Script | Input | Output |
|------|--------|-------|--------|
| NFP test (avg-frames) | `jobs/videomae_sae/run_nfp_test_avg_tau.sh` | NFP dataset + SAE from 1a | `/net/scratch2/renqy/nfp_results/videomae_deadpen_0p03_avgtau.pt` |

```bash
sbatch jobs/videomae_sae/run_nfp_test_avg_tau.sh
```

**Results:** 151/6144 significant (2.46%). Accel_mag increases most (0.59% → 2.10%),
consistent with avg-frames smoothing reducing tubelet-boundary noise for derived
quantities. Selectivity matrix remains diagonal-dominant.

**Notes:**
- SAE: expansion ×8 → 6144 features, l1=0.1, dead_penalty=0.03, 15,320 steps, batch 4096.
- Tau modes: `first_frame` (default, primary result) assigns each tubelet the tau of its
  first constituent frame; `avg_frames` averages over both frames. See `--tau_mode` arg
  in `analysis/nfp_test.py`.
- DINOv2 and synthetic controls were run under `first_frame` mode; VideoMAE first-frame
  (75 features) is the correct baseline for cross-condition comparisons.
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

**Results:** SAE: 0.475 ± 0.063. Raw baseline: 0.469 ± 0.007.

**Notes:**
- Both scripts call `eval/metric.py` (not `metric.py` — correct path relative to project root).
- Prerequisites: DINOv2 embeddings must exist (run `jobs/dataset/run_ssv2_dino_embeddings.sh`).
- `eval_mono_auxk.sh` also requires VideoMAE SAE checkpoint.

---

## 5b. Monosemanticity Score — PCA / ICA baselines (`jobs/monosemanticity/`)

**What it tests:** Computes the same MS score for **PCA** and **ICA** decompositions
of the VideoMAE layer-11 representation, as additional "filter" baselines alongside
the SAE and the raw activations. This isolates how much of the SAE's interpretability
is specific to sparse, overcomplete coding versus being achievable by any linear
decomposition (orthogonal/variance-maximizing for PCA, statistically-independent for ICA).

**Paper sections:** §3.3 (MS comparison table)

PCA and ICA are wrapped in the SAE `Dictionary` interface (`PCADict` / `ICADict` in
`dictionary_learning/dictionary.py`), so the extraction (`save_activations.py`) and
scoring (`eval/metric.py`) stages are reused unchanged — guaranteeing an apples-to-apples
comparison (identical clips, DINOv2 embeddings, per-token → max-pool aggregation, metric).

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Fit PCA + ICA | `jobs/monosemanticity/fit_pca_ica.sh` | SAE train activations `/net/scratch/renqy/activations/train` | `/net/scratch2/renqy/linear_decomp/{pca,ica}.pt` |
| 2. MS for PCA | `jobs/monosemanticity/eval_mono_pca.sh [sign_split\|abs]` | `pca.pt`, DINOv2 embeddings | MS under `sae_activations/pca_<mode>_val/ms_dinov2/` |
| 3. MS for ICA | `jobs/monosemanticity/eval_mono_ica.sh [sign_split\|abs]` | `ica.pt`, DINOv2 embeddings | MS under `sae_activations/ica_<mode>_val/ms_dinov2/` |

```bash
sbatch jobs/monosemanticity/fit_pca_ica.sh           # once; produces pca.pt + ica.pt
sbatch jobs/monosemanticity/eval_mono_pca.sh         # sign_split (primary, 1536 features)
sbatch jobs/monosemanticity/eval_mono_ica.sh
sbatch jobs/monosemanticity/eval_mono_pca.sh abs     # |proj| robustness (768 features)
sbatch jobs/monosemanticity/eval_mono_ica.sh abs
```

**Sign handling (signed components → non-negative MS metric).** PCA/ICA components are
signed, but the MS metric assumes non-negative "feature present" activations. Primary
mode `sign_split` maps each component `c → [ReLU(c), ReLU(-c)]` (768 → 1536 half-features),
matching ReLU-SAE semantics and removing ICA's arbitrary sign (precedent: Cunningham et al.
2023; ICA-Lens 2026). Robustness mode `abs` uses `|c|` (768 features). Selected at eval
time via `--decomp_mode`; no refit needed.

**Notes:**
- **Dimensionality asymmetry:** linear PCA/ICA on 768-dim activations yield ≤768 components
  (≤1536 after sign-split) vs the SAE's 6144 overcomplete features. This cap is inherent and
  should be noted; MS is averaged per-feature (a density, not a total), so the comparison is
  still fair. Compare *distributions* (peak + spread), not just means — the SAE's edge is its
  high-MS tail (peak 0.80 vs raw max 0.49), not its mean.
- PCA and ICA are fit on the **same** 500k-vector random subsample of the SAE training corpus
  for symmetry. FastICA whitens internally (`whiten='unit-variance'`). If the fit log reports
  ICA non-convergence, re-run `fit_pca_ica.sh` with fewer components (e.g. `--n_components 256`)
  or larger `--ica_max_iter`.
- New code: `dictionary_learning/dictionary.py` (`PCADict`/`ICADict`), `analysis/fit_pca_ica.py`,
  and the `--sae_model pca|ica` + `--decomp_mode` branches in `training/extract_activations.py`
  (deployed to the cluster as `save_activations.py`). `eval/metric.py` and `models/videomae.py`
  are unchanged.
- Prerequisites: DINOv2 embeddings (`run_ssv2_dino_embeddings.sh`) and SAE training activations
  (`run_extract_videomae_activations.sh`) must already exist.

**Local single-GPU alternative (no cluster):** `jobs/local/run_pca_ica_ms_local.ps1` runs the
entire fit → DINOv2-embed → encode → MS chain on a local machine with the full SSv2 dataset and
a CUDA GPU. It extracts a small fitting corpus locally (the cluster's 44M-token corpus is not
needed) and writes everything under `local_runs/`. See the "Local execution" section at the top
of this file.

**Results (local, 256 components → 512 sign-split features, fit on ~600k tokens / 400 SSv2-train
videos, scored on 800 val clips):** PCA **0.467 ± 0.006** (peak 0.497), ICA **0.467 ± 0.009**
(peak 0.510), vs SAE **0.475 ± 0.063** (peak 0.802) and raw **0.469 ± 0.007** (peak 0.490).
PCA/ICA/raw all cluster at mean ~0.467–0.469 with tight spread and peaks ~0.50; only the SAE has
a high-MS tail (peak 0.80). The local numbers are the valid *relative* comparison (component
count and fit corpus differ from a full-rank cluster run).

---

## 5c. NFP test — raw layer / PCA / ICA baselines (`jobs/local/run_nfp_local.ps1`)

**What it tests:** Runs the no-false-positives (NFP) temporal-feature test on the **raw layer-11
dimensions**, **PCA**, and **ICA** — the NFP analog of the §5b MS baselines. The no-false-positive
guarantee is a property of the within-video covariance statistic + stimulus design, **not** of the
SAE, so it holds for any feature basis. This isolates the SAE's contribution: sparsity/selectivity.

**Paper sections:** §4 (NFP results — baseline comparison)

`analysis/nfp_test.py` takes `--sae_model {standard,pca,ica,identity}` and `--decomp_mode`.
`identity` uses `IdentityDict` (encode = identity) → the 768 raw dimensions as features.
PCA/ICA reuse the `pca.pt`/`ica.pt` fit in §5b. The covariance statistic does **not** require
non-negativity (unlike MS), so signed components are valid; `sign_split` is used to keep
"feature" semantics parallel to the ReLU SAE.

```powershell
# Local (reuses local_runs/decomp/{pca,ica}.pt from run_pca_ica_ms_local.ps1):
powershell -ExecutionPolicy Bypass -File jobs\local\run_nfp_local.ps1
# Cluster equivalent (run analysis/nfp_test.py directly per condition):
python analysis/nfp_test.py --dataset_dir <nfp_dir> --sae_model identity \
    --output_path raw_nfp.pt --label "Raw layer (768 dims)"
python analysis/nfp_test.py --dataset_dir <nfp_dir> --sae_model pca \
    --sae_path pca.pt --decomp_mode sign_split --output_path pca_nfp.pt --label "PCA"
python analysis/nfp_test.py --dataset_dir <nfp_dir> --sae_model ica \
    --sae_path ica.pt --decomp_mode sign_split --output_path ica_nfp.pt --label "ICA"
```

**Results (local, 3000 ball videos, features significant for ≥1 tau at Bonferroni p<0.05/D):**

| Condition | Features D | Significant ≥1 τ | Diagonal-dominant | Non-sig mean \|C\| |
|---|---|---|---|---|
| Raw layer | 768 | 698 (90.9%) | No (direction row peaks at v_y) | 1.5×10⁻² |
| PCA (sign_split) | 512 | 351 (68.6%) | Yes | 7.7×10⁻³ |
| ICA (sign_split) | 512 | 352 (68.8%) | Yes | 8.4×10⁻⁴ |
| SAE (cluster ref) | 6144 | 75 (1.2%) | Yes | 1.5×10⁻⁴ |

**Interpretation:** every flagged feature is a *genuine* temporal encoder (guarantee holds for all
bases), but raw/PCA/ICA flag 69–91% of their features as temporal — the signal is smeared across
the basis and is not localizable. Only the SAE's sparsity isolates a small (1.2%) interpretable
set. Raw additionally fails selectivity (not diagonal-dominant); PCA/ICA recover selectivity but
remain dense. So the SAE's NFP value is **sparsity + selectivity**, not the guarantee itself.

---

## 5d. PCA/ICA dimensionality sweep (`analysis/sweep_pca_ica_dim.py`)

**What it tests:** How MS and the NFP significant-fraction vary with the number of PCA/ICA
components D. Confirms the SAE's advantages are not recoverable by tuning D.

**Paper sections:** §3.3 / §4 (sensitivity analysis)

Efficient design: the VideoMAE forward passes are independent of D, so they are cached **once**
(per-token val acts for MS; ball-tracking acts for NFP), then D is swept as linear algebra. PCA
is nested (fit once at the largest D, slice); ICA is refit per D (not nested). The fit is
**mode-independent**, so `--modes` evaluates several sign-handling modes per (D, method) for free.

```powershell
# convenience runner (grid to full-rank 768, both modes, fixed-768 cutoff):
powershell -ExecutionPolicy Bypass -File jobs\local\run_sweep_local.ps1
# or directly:
$env:PYTHONPATH = (Get-Location).Path
python analysis/sweep_pca_ica_dim.py `
    --ssv2_path <SSv2> --nfp_dir data/output/nfp --train_dir local_runs/train_acts `
    --embeds_path local_runs/embeds/ssv2_val_dinov2.pt --output_csv local_runs/sweep_dim_fixed768.csv `
    --grid 16 32 64 128 256 512 768 --methods pca ica --modes sign_split signed `
    --fixed_denom 768 --device cuda:0
```

The CSV columns are `D, method, mode, n_features, ms_mean, ms_std, ms_peak, nfp_sig, nfp_pct,
nfp_sig_fixed768, nfp_pct_fixed768, diag_dominant`. The last two report NFP significance under a
**D-independent** Bonferroni cutoff (`α/768`, denominator fixed at the raw layer dim) alongside
the adaptive `α/M` count, so significance can be compared across D without the changing
multiple-comparisons correction as a confound. Canonical CSV: `results/pca_ica_baselines/
sweep_dim_fixed768.csv` (per-mode splits: `sweep_dim_sign_split.csv` / `sweep_dim_signed.csv`).

**Results — sign_split (`local_runs/sweep_dim.csv`):** MS mean is flat (~0.467) at every D for
both methods; MS std shrinks with D (0.012→0.005); MS **peak** is flat (PCA ~0.497 — identical at
every D because PCA is nested; ICA ~0.51–0.52) and never approaches the SAE's 0.80. NFP
%-significant stays dense at every D (PCA 59%→69%, ICA 62%→70%, already ~60% at D=16) and never
approaches the SAE's 1.2%. FastICA blew up numerically at D=512 (NaNs — high-D instability;
PCA-512 is fine and the script records the ICA failure as a NaN row instead of crashing).

**Results — signed robustness (`local_runs/sweep_dim_signed.csv`):** the conclusions are robust
to sign handling. MS mean is again flat ~0.467 at every D; peak capped (PCA 0.484 identical at
every D; ICA ~0.48–0.51), never near 0.80. NFP %-significant is **even denser** under signed
(PCA 83–91%, ICA 91–94%) than under sign_split — approaching the raw-layer 90.9%, consistent with
signed full-rank PCA being a rotation of the raw basis. Signed features are mostly *not*
diagonal-dominant (less concept-selective), which is why `sign_split` is the better primary mode
while `signed` serves as the robustness check. (ICA-512 fails the same way in signed mode.)

**Results — full rank (D=768) + fixed cutoff (`results/pca_ica_baselines/sweep_dim_fixed768.csv`):**
full-rank PCA does not help — at D=768 it still flags 68.8% (sign_split) / 90.9% (signed), never
near the SAE's 1.2%. The denseness is robust to the cutoff: swapping the adaptive `α/M` bar for
the fixed `α/768` bar moves the fractions only a few points (PCA/ICA stay in ~47–91% at every D
under either threshold). FastICA NaNs at **both** D=512 and D=768. Internal check: PCA-signed at
D=768 gives **698/768 — exactly the raw layer**, since full-rank signed PCA is a rotation of the
raw basis and the per-feature t-test count is rotation-invariant.

**Conclusion:** under both sign modes, all D up to full rank, and either Bonferroni cutoff, the
SAE's high-MS tail (peak 0.80) and NFP sparsity (1.2%) are intrinsic to sparse overcomplete coding
— not recoverable by choosing the right D, sign handling, or significance threshold for PCA/ICA.

**Note on the NFP detection threshold:** a feature is "significant" if its within-video
covariance with some tau has a one-sample *t*-test (over N=3000 videos) `p < 0.05/D` (Bonferroni
across features). This is a *significance*, not effect-size, threshold — with N=3000 the test is
high-powered, so even tiny non-zero covariances pass. The high PCA/ICA/raw densities reflect that
nearly every linear direction has a small but genuine motion covariance; the SAE's sparsity (not a
stricter threshold) is what isolates the few strong temporal features.

## 5e. PCA/ICA on the DINO negative + synthetic positive controls

**What it tests:** whether the linear-decomposition baselines behave on the two *controls* the
way the SAE does — i.e. whether the PCA/ICA story (sparse SAE vs dense linear bases) is confirmed
where we have an unambiguous expected answer. The same `PCADict`/`ICADict` drop-in is fit on each
control's own representations and run through that control's NFP test.

**Paper sections:** §2 (negative control) / §3–§4 (positive control).

> MS does not transfer to either control: on DINO it would be circular (DINOv2 *is* the MS
> image encoder); on synthetic there are no natural frames. For synthetic the stronger analog is
> **ground-truth `W_τ` projection-fraction** alignment (already implemented in
> `nfp_test_synthetic.py`); for DINO the test is purely NFP = 0.

**DINO negative control** — `jobs/local/run_dino_pca_ica.ps1`: extract DINOv2 spatial patch
activations over an independent SSv2-train corpus → fit PCA/ICA (256 comps) → run
`analysis/nfp_test_dino_patch.py --sae_model {pca,ica,identity}` on the ball dataset. **Result:
every basis flags 0 temporal features** — SAE 0/6144, raw 0/768, PCA 0/512, ICA 0/512 (ICA
converges here, 66 iters). This is the load-bearing control: the same PCA/ICA recipe that flags
69–91% on VideoMAE flags **nothing** on a temporally-blind encoder, so the VideoMAE flags are real
temporal content, not an artifact of applying NFP to a non-sparse basis. Details:
`results/pca_ica_baselines/dino_negcontrol.md`.

**Synthetic positive control (100- and 763-static)** — `jobs/local/run_synth_pca_ica.ps1`: dump
the synthetic reps (`analysis/dump_synth_acts.py`) → fit PCA/ICA (256 comps) → run
`analysis/nfp_test_synthetic.py --sae_model {pca,ica}` (NFP + `W_τ`/`W_static` alignment), plus a
same-pipeline SAE reference. **Result (100-static):** SAE is sparse (31/6144 = 0.50%) and its
flagged features genuinely load on the temporal subspace (ProjFrac `W_τ` 0.229, ~35× random);
PCA is dense (41%) and its flags are static-dominated (`W_τ` 0.048, `W_static` 0.952), so it does
*not* isolate the temporal subspace; **ICA did not converge** (FastICA iteration cap) and is
reported as such. **763-static:** `W_static` spans 763/768 dims so the alignment metric saturates,
but the sparsity gap is starker — SAE 0.50% vs PCA *and* ICA both 100%. Details:
`results/pca_ica_baselines/synthetic_alignment.md`.

**Dimensionality sweep on both controls** (mirrors §5d) — `jobs/local/run_control_sweeps.ps1`,
via `analysis/sweep_dino_dim.py` (NFP-only) and `analysis/sweep_synth_dim.py` (NFP + `W_τ`
alignment); shared helpers in `analysis/sweep_common.py`. CSVs: `sweep_dim_dino.csv`,
`sweep_dim_synth{100,763}.csv`. **DINO: 0 significant at every D** (the two stray single-feature
blips under the adaptive bar vanish under the fixed α/768 cutoff) — the negative control is robust
across all dimensionalities. **Synthetic-100:** PCA's absolute flagged count **saturates at ~210**
(nested PCA — extra components are non-temporal noise), so %sig falls with D only via the growing
denominator while `W_τ` alignment stays flat (~0.048, static-dominated); ICA NaNs at D≥512.
**Synthetic-763:** 100% at every D. So, as in the VideoMAE sweep, no choice of D moves PCA/ICA
toward the SAE on either control.

**Conclusion:** both controls confirm the main study. The negative control shows NFP has no
basis-dependent false positives (every basis → 0 on DINO, at every D); the positive control shows,
against ground truth, that only the SAE is both sparse and correctly localized on the temporal
directions, at every D.
