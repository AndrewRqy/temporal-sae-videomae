# PCA / ICA baselines — consolidated results summary

All headline tables for the linear-decomposition baseline PR in one place.
Per-condition NFP selectivity (diagonal) matrices are in `nfp_selectivity.md`;
per-feature MS lists in `ms_{pca,ica}_sign_split.txt`; raw sweep data in the CSVs.
Local single-GPU numbers; SAE column is the cluster reference.

## 1. Monosemanticity (MS)

| Filter | Features | MS mean ± std | Peak | Diag-dominant (NFP) |
|---|---|---|---|---|
| **SAE** | 6144 | **0.475 ± 0.063** | **0.802** | Yes |
| Raw layer | 768 | 0.469 ± 0.007 | 0.490 | No |
| PCA (sign_split) | 512 | 0.467 ± 0.006 | 0.497 | Yes |
| ICA (sign_split) | 512 | 0.467 ± 0.009 | 0.510 | Yes |

## 2. NFP significant features + diagonal dominance

| Filter | Features | Significant ≥1 τ | Diagonal-dominant | Non-sig mean \|C\| |
|---|---|---|---|---|
| **SAE** (cluster ref) | 6144 | **75 (1.2%)** | Yes | 1.5×10⁻⁴ |
| Raw layer | 768 | 698 (90.9%) | No | 1.5×10⁻² |
| PCA (sign_split) | 512 | 351 (68.6%) | Yes | 7.7×10⁻³ |
| ICA (sign_split) | 512 | 352 (68.8%) | Yes | 8.4×10⁻⁴ |

Full per-tau counts and the 5×5 selectivity (diagonal) matrices: `nfp_selectivity.md`.

## 3. Dimensionality sweep — MS, NFP, and diagonal dominance vs D

### sign_split mode

| D | method | feats | MS mean | MS peak | NFP %sig | diag-dominant |
|---|---|---|---|---|---|---|
| 16 | pca | 32 | 0.4677 | 0.497 | 59.38% | False |
| 16 | ica | 32 | 0.4677 | 0.511 | 62.50% | False |
| 32 | pca | 64 | 0.4675 | 0.497 | 59.38% | True |
| 32 | ica | 64 | 0.4675 | 0.512 | 65.62% | True |
| 64 | pca | 128 | 0.4666 | 0.497 | 59.38% | False |
| 64 | ica | 128 | 0.4672 | 0.515 | 68.75% | False |
| 128 | pca | 256 | 0.4669 | 0.497 | 66.02% | True |
| 128 | ica | 256 | 0.4676 | 0.519 | 67.97% | True |
| 256 | pca | 512 | 0.4672 | 0.497 | 66.60% | True |
| 256 | ica | 512 | 0.4674 | 0.512 | 70.51% | True |
| 512 | pca | 1024 | 0.4677 | 0.497 | 69.34% | True |
| 512 | ica | 1024 | nan | nan | nan% | False |

### signed mode (robustness)

| D | method | feats | MS mean | MS peak | NFP %sig | diag-dominant |
|---|---|---|---|---|---|---|
| 16 | pca | 16 | 0.4658 | 0.484 | 87.50% | True |
| 16 | ica | 16 | 0.4686 | 0.484 | 93.75% | True |
| 32 | pca | 32 | 0.4642 | 0.484 | 87.50% | False |
| 32 | ica | 32 | 0.4669 | 0.495 | 93.75% | False |
| 64 | pca | 64 | 0.4645 | 0.484 | 82.81% | False |
| 64 | ica | 64 | 0.4659 | 0.500 | 90.62% | False |
| 128 | pca | 128 | 0.4657 | 0.484 | 88.28% | False |
| 128 | ica | 128 | 0.4667 | 0.514 | 91.41% | True |
| 256 | pca | 256 | 0.4667 | 0.484 | 89.84% | False |
| 256 | ica | 256 | 0.4673 | 0.502 | 91.02% | False |
| 512 | pca | 512 | 0.4675 | 0.484 | 91.02% | False |
| 512 | ica | 512 | NaN | NaN | NaN | False |

### Diagonal-dominance summary (✓ = diagonal is row-max in every populated row)

| D | PCA sign_split | ICA sign_split | PCA signed | ICA signed |
|---|---|---|---|---|
| 16 | ✗ | ✗ | ✓ | ✓ |
| 32 | ✓ | ✓ | ✗ | ✗ |
| 64 | ✗ | ✗ | ✗ | ✗ |
| 128 | ✓ | ✓ | ✗ | ✓ |
| 256 | ✓ | ✓ | ✗ | ✗ |
| 512 | ✓ | ✗ | ✗ | ✗ |

Diagonal dominance is **mode-dependent**: under `sign_split` it generally holds at D≥128
(PCA 128/256/512 ✓; ICA 128/256 ✓, 512 fails numerically — FastICA NaN), consistent with
sign-split features being more concept-selective. Under `signed` it mostly fails — another
reason `sign_split` is the primary mode and `signed` is the robustness check. The pattern is
noisy at small D (too few features to be selective). The per-condition 5×5 matrices behind
these flags are in `nfp_selectivity.md`.
