# Monosemanticity (MS) scores — SAE vs PCA / ICA / raw

Weighted-pairwise-cosine MS over 800 SSv2-val clips with DINOv2 embeddings
(higher = more interpretable). PCA/ICA use 256 components -> 512 sign-split features.

| Filter | Features | MS mean +/- std | Peak | Dead | Source |
|---|---|---|---|---|---|
| **SAE (local)** | 6144 | **0.468 +/- 0.065** | **0.782** | 56 | local (ms_sae_local.txt) |
| SAE (cluster, ref) | 6144 | 0.475 +/- 0.063 | 0.802 | 5 | cluster (results/ms_standard_deadpen.txt) |
| Raw layer | 768 | 0.469 +/- 0.007 | 0.490 | 0 | cluster (results/mono_raw_872227.txt) |
| PCA (sign_split) | 512 | 0.467 +/- 0.006 | 0.497 | 0 | local (ms_pca_sign_split.txt) |
| ICA (sign_split) | 512 | 0.467 +/- 0.009 | 0.510 | 0 | local (ms_ica_sign_split.txt) |

The local SAE (run through the same pipeline as PCA/ICA/raw) reproduces the cluster SAE
(0.468 vs 0.475; peak 0.78 vs 0.80). PCA/ICA/raw cluster at mean ~0.467-0.469 with a tight
spread and peaks ~0.50; only the SAE has a high-MS tail (peak ~0.78-0.80). Full per-feature
top/bottom-10 lists: `ms_sae_local.txt`, `ms_pca_sign_split.txt`, `ms_ica_sign_split.txt`.

MS-vs-D (both sign modes) is in `sweep_dim_sign_split.csv` / `sweep_dim_signed.csv`.
