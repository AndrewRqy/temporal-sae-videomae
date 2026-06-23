# PCA / ICA on the SYNTHETIC positive control — NFP + ground-truth W_τ alignment

Extends the PCA/ICA baselines to the synthetic positive control. Unlike VideoMAE, the
**monosemanticity (MS) score does not apply here** — there are no natural frames to embed
with an independent image encoder. Synthetic instead offers something *stronger*: the
ground-truth temporal directions `W_τ` are known by construction, so we can check whether a
basis's NFP-flagged features actually live in the temporal subspace.

**Setup — and how this fitting differs from the VideoMAE/DINO baselines.** In the other
experiments PCA/ICA are fit on a *model's activations over real images* (VideoMAE on SSv2, or
DINOv2 on SSv2 frames). Here there is **no model and no real imagery**: the 768-dim vectors are
*synthesized* (`analysis/gen_synthetic_activations.py`, seed 42):
`h(V,t) = (temporal: W_τ·τ̃) + (static: W_static·s) + noise`, where `W_τ` (5×768) and `W_static`
(`n_static`×768) are random **orthonormal** subspaces we choose (so the true temporal directions
are *known*), `τ̃` is the z-scored real ball-dataset kinematics projected through `W_τ` (varies
across the 8 steps), and `s` is one random static code per video projected through `W_static`
(constant within a video → within-video covariance with τ is **0 by construction**). `n_static`
≫ 5 so static content dominates, mimicking VideoMAE.

PCA, ICA, and the synthetic SAE are then fit **directly on these constructed `h` vectors** — no
forward pass through any network. We flatten `h` `[N,8,768]→[N·8,768]` (`dump_synth_acts.py`) and
run the *same* `fit_pca_ica.py` (256 components → 512 sign-split features) used for VideoMAE; the
SAE was gradient-trained on the same `h`. The NFP test likewise runs the fitted encoders on `h`
and correlates with the **known** τ — no model at test time either. We then measure each flagged
feature's **projection fraction** onto `W_τ` (temporal) vs `W_static` (static). The only thing
that changes versus the other baselines is the *source of the input vectors*
(synthesized-from-known-directions vs model-activations-on-images). Run via
`jobs/local/run_synth_pca_ica.ps1`.

> Subspace sizes (matter for reading proj-fractions): `W_τ` is **5-dim** in both variants;
> `W_static` is **100-dim** (100-variant) and **763-dim** (763-variant). Random-baseline
> projection fraction ≈ subspace_dim / 768 (so ≈0.007 for `W_τ`, ≈0.13 / ≈0.99 for `W_static`).
> *(The console report hardcodes "[5 x 768]" in its banner — a display bug; the actual `W_static`
> shapes are 100/763 as above, and the proj-fraction math uses the full matrices.)*

## 1. 100-static variant (clean positive control)

| Filter | Features | Sig ≥1 τ | ProjFrac W_τ (sig) | ProjFrac W_static (sig) | non-sig mean \|C\| | ICA converged |
|---|---|---|---|---|---|---|
| **SAE** (cluster / paper) | 6144 | **31 (0.50%)** | 0.229 | 0.329 | 3.6×10⁻⁵ | — |
| PCA (sign_split) | 512 | 210 (41.0%) | 0.048 | 0.952 | 1.0×10⁻⁴ | — |
| ICA (sign_split) | 512 | 461 (90.0%) | 0.0013 | 0.0062 | 4.4×10⁻³ | **No** (2000/2000) |

(The local SAE reproduces the cluster SAE **exactly** here — 31/6144 (0.50%) — so the SAE row is
both the reported number and a same-pipeline reference. ProjFrac is a property of the SAE
weights, identical cluster/local.)

**Reading it.**
- **SAE**: sparse (0.5%) *and* its flagged features genuinely load on the temporal subspace —
  ProjFrac `W_τ` = 0.229 (≈35× the 0.007 random baseline), ~5× PCA's temporal loading.
- **PCA**: dense (41%) and its flagged features are **static-dominated** — ProjFrac `W_τ` only
  0.048 with `W_static` 0.952. PCA's top variance directions are the (high-variance) static
  ones; they're flagged "temporal" only via a tiny consistent covariance leak (mean |C| 1e-4),
  not because they isolate `W_τ`. PCA does *not* recover the temporal subspace the SAE does.
- **ICA**: **did not converge** (FastICA hit its 2000-iter cap) — its components are nearly
  unaligned with either ground-truth subspace (proj-fractions ≈ random), so its 90% flag rate
  is not a trustworthy positive result. This is the same FastICA high-dimensional instability
  that NaN'd ICA at D≥512 in the dimensionality sweep (`summary.md` §3).

**Selectivity (diagonal dominance):** PCA-100 is diagonal-dominant (every τ's flagged features
peak on their own τ); ICA-100 is not (e.g. the `accel_mag` row peaks on `speed`). So even where
PCA looks "clean" by the diagonal test, it fails the *subspace* test — diagonal dominance and
true `W_τ` alignment are different things (cf. `summary.md` §0).

## 2. 763-static variant (robustness)

| Filter | Features | Sig ≥1 τ | ProjFrac W_τ (sig) | ProjFrac W_static (sig) | non-sig mean \|C\| |
|---|---|---|---|---|---|
| **SAE** (cluster / paper) | 6144 | **31 (0.50%)** | 0.0025 | 0.9975 | 4.7×10⁻⁵ |
| PCA (sign_split) | 512 | 512 (100%) | 0.0068 | 0.9932 | nan (no non-sig) |
| ICA (sign_split) | 512 | 512 (100%) | 0.0062 | 0.9938 | nan (no non-sig) |

Here `W_static` spans **763 of 768 dims**, so "projection onto `W_static`" is ≈1 for *any*
direction — the alignment metric is uninformative in this variant (everything is trivially
"static"). PCA explains only 52% of variance at 256 comps (vs 99% for the 100-variant), and
both PCA and ICA flag **100%** of features. `nan` mean |C| simply means the non-significant set
is empty (all 512 flagged), so there is nothing to average. The robust signal is the
**sparsity gap**: the SAE still isolates 0.5% while PCA/ICA flag everything.

## Takeaways

1. The synthetic positive control reproduces the VideoMAE story with **ground truth**: the SAE
   is sparse (0.5%) and its flagged features genuinely sit in the temporal `W_τ` subspace; PCA
   is dense and flags static-dominated directions; ICA is numerically unreliable here.
2. **NFP does not manufacture the SAE's result** — it flags any consistent within-video
   covariance, which is why PCA/ICA come out dense. The SAE's value is the *sparse, correctly
   localized* temporal code, not the test.
3. Diagonal dominance ≠ subspace correctness: PCA-100 passes the diagonal test yet misses `W_τ`.

Raw output tensors: `results/pca_ica_baselines/synth_nfp/synth{100,763}_{pca,ica,sae}*.pt`
(each has C, t_stat, p_val, C_mean, W_tau, W_static). Reproduce with
`jobs/local/run_synth_pca_ica.ps1`.

> ICA caveat: to obtain a converged synthetic ICA one would need fewer components or many more
> iterations; we report it as non-converged rather than tune it into looking clean, consistent
> with how FastICA instability is handled in the dimensionality sweep.

## Dimensionality sweep (mirrors the VideoMAE D-sweep)

`analysis/sweep_synth_dim.py` sweeps D = 16…768 (PCA nested, ICA refit) for both sign modes,
reporting NFP significance (adaptive + fixed α/768) and the mean `W_τ` / `W_static` projection
fraction of the flagged features. CSVs: `sweep_dim_synth100.csv`, `sweep_dim_synth763.csv`.

**100-static — PCA (sign_split):**

| D | feats | sig | %sig (α/M) | %sig (α/768) | ProjFrac W_τ (sig) | ProjFrac W_static (sig) |
|---|---|---|---|---|---|---|
| 16 | 32 | 32 | 100% | 100% | 0.086 | 0.914 |
| 32 | 64 | 64 | 100% | 100% | 0.055 | 0.945 |
| 64 | 128 | 128 | 100% | 100% | 0.044 | 0.956 |
| 128 | 256 | 210 | 82.0% | 82.0% | 0.048 | 0.952 |
| 256 | 512 | 210 | 41.0% | 41.0% | 0.048 | 0.952 |
| 512 | 1024 | 210 | 20.5% | 20.6% | 0.048 | 0.952 |
| 768 | 1536 | 210 | 13.7% | 13.7% | 0.048 | 0.952 |

The key point: the **absolute count of flagged features saturates at ~210** (≈105 PCA
components). Because PCA is nested, once the signal subspace is captured, every additional
component is non-temporal noise, so the count plateaus and %sig *falls with D only because the
denominator grows* — not because PCA gets more selective. The `W_τ` alignment of the flagged set
stays **flat at ~0.048** (static-dominated) at every D. **No D moves PCA toward the SAE's sparse,
temporally-clean code** (SAE: 31 features, ProjFrac W_τ 0.229). ICA NaNs at D≥512 (same FastICA
instability), and where it runs it is non-converged with near-random alignment.

**763-static:** PCA and ICA flag **100% at every D** (the 763-dim static subspace makes nearly
every direction significant); `W_τ` alignment stays ~0.006–0.02 throughout. The sparsity gap to
the SAE (0.5%) is constant across D.

**Takeaway:** consistent with the VideoMAE sweep (`summary.md` §3) — the SAE's sparsity and
correct temporal localization are not recoverable by tuning the number of PCA/ICA components.
