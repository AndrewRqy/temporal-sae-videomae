# PCA / ICA on the DINOv2 NEGATIVE control — NFP test

Extends the PCA/ICA baselines to the DINOv2 negative control, mirroring the VideoMAE setup:
fit the decomposition on an **independent** corpus (SSv2-train DINO patch activations), then
run the NFP within-video covariance test on the 3000-video ball dataset, reading off the
ball-containing spatial patch token at each step. DINOv2 processes **each frame independently**
(no temporal context), so the expected result for *every* basis is **0 temporal features**.

MS does not apply here (DINOv2 is itself the encoder used for the MS image-similarity matrix —
running MS on it would be circular). The negative control is purely the NFP test.

Fit: 256 components (→ 512 sign-split features) on ~600k SSv2-train DINO patch tokens.
Run via `jobs/local/run_dino_pca_ica.ps1`.

## Results — significant temporal features (Bonferroni p < α/D)

| Filter | Features | Significant ≥1 τ | Overall mean \|C\| | ICA converged |
|---|---|---|---|---|
| **SAE** (cluster / paper) | 6144 | **0 (0.00%)** | — | — |
| Raw DINO patch (identity) | 768 | **0 (0.00%)** | 2.95×10⁻³ | — |
| PCA (sign_split) | 512 | **0 (0.00%)** | 2.11×10⁻³ | — |
| ICA (sign_split) | 512 | **0 (0.00%)** | 9.61×10⁻⁴ | Yes (66 iters) |

Bonferroni thresholds: PCA/ICA p < 9.77×10⁻⁵ (= 0.05/512); identity p < 6.51×10⁻⁵ (= 0.05/768).

## Takeaways

1. **The negative control holds for every basis.** SAE, PCA, ICA, and the raw DINO patch
   dimensions all flag **0** temporal features. DINOv2 has no temporal context, so there is no
   temporal structure for *any* decomposition to find.
2. **This is the key control for the whole baseline study.** On VideoMAE, PCA/ICA/raw flag
   69–91% of features as temporal (`summary.md` §2). One might worry that dense flagging is an
   artifact of applying NFP to a non-sparse basis. The DINO result rules that out: the *same*
   PCA/ICA recipe on a temporally-blind encoder flags **nothing**. So the VideoMAE flags
   reflect genuine temporal content in VideoMAE — NFP has no false positives regardless of
   basis; what differs across bases is only how *sparsely* the real signal is localized.
3. **ICA converged here** (66 iterations), unlike the synthetic reps and high-D VideoMAE sweep
   where FastICA blew up. The ICA instability is data-/dimensionality-dependent, not a blanket
   failure — and where it does run, it still returns the correct 0 for this control.

Raw output tensors: `results/pca_ica_baselines/dino_nfp/dino_{pca,ica}_sign_split.pt`,
`dino_identity.pt` (each has C, t_stat, p_val, C_mean). Reproduce with
`jobs/local/run_dino_pca_ica.ps1`.

## Dimensionality sweep (mirrors the VideoMAE D-sweep)

`analysis/sweep_dino_dim.py` caches the DINOv2 ball-token acts once and sweeps
D = 16…768 (PCA nested, ICA refit) for both sign modes, NFP-only (MS is circular here).
CSV: `sweep_dim_dino.csv`.

**Result: 0 significant features at every D**, for PCA and ICA, sign_split and signed. Two
isolated single-feature blips appear under the *adaptive* bar (ICA signed D=32 → 1; ICA
sign_split D=128 → 1) and both **vanish under the fixed α/768 cutoff** → 0. So the negative
control is robust across the whole dimensionality range: no number of components manufactures a
temporal feature from a temporally-blind encoder. (This is the across-D version of the headline
table above.)
