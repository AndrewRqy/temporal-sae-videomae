# NFP test — selectivity (diagonal) scores: SAE / raw / PCA / ICA  (all local)

Within-video covariance + one-sample t-test over 3000 ball videos. A feature is
significant for tau k if p < 0.05/D (Bonferroni). Selectivity matrix: for features
significant in the ROW tau, the mean |t-stat| across each COLUMN tau; the diagonal
(bold *) should be the row max if features are concept-selective.

All four conditions computed on the SAME local pipeline, so the SAE/raw/PCA/ICA matrices
are directly comparable here. NOTE: the **SAE row below is the local pipeline-validation run**
(85/6144), included only so all four sit on one pipeline — it is *not* the reported SAE result.
The reported SAE numbers are the cluster/paper ones (75/6144 = 1.22%); see `summary.md` §0.
The local SAE reproduces the cluster SAE closely (85 vs 75), which is the point of showing it.

## SAE (6144 features, local)   (D=6144, threshold p<8.14e-06)

| tau | sig+ | sig- | total% |
|---|---|---|---|
| speed | 15 | 25 | 0.65% |
| vel_x | 16 | 24 | 0.65% |
| vel_y | 13 | 11 | 0.39% |
| accel_mag | 3 | 10 | 0.21% |
| direction | 8 | 7 | 0.24% |

**Significant for >=1 tau: 85/6144 (1.38%)** | non-sig mean |C| = 2.366e-04

Selectivity matrix (mean |t| for sig-in-row; * = diagonal is row max):

| sig in \ measured | speed | vel_x | vel_y | accel_mag | direction |
|---|---|---|---|---|---|
| speed | **6.19*** | 3.22 | 2.98 | 2.71 | 2.40 |
| vel_x | 3.67 | **6.32*** | 2.76 | 2.26 | 2.21 |
| vel_y | 4.02 | 3.55 | **6.19*** | 2.53 | 4.88 |
| accel_mag | 4.99 | 2.41 | 2.54 | **5.67*** | 2.62 |
| direction | 3.99 | 2.75 | 6.54 | 2.99 | **6.24*** |

**Diagonal-dominant: False**

## Raw layer (768 dims, identity)   (D=768, threshold p<6.51e-05)

| tau | sig+ | sig- | total% |
|---|---|---|---|
| speed | 145 | 147 | 38.02% |
| vel_x | 200 | 201 | 52.21% |
| vel_y | 164 | 147 | 40.49% |
| accel_mag | 213 | 168 | 49.61% |
| direction | 148 | 143 | 37.89% |

**Significant for >=1 tau: 698/768 (90.89%)** | non-sig mean |C| = 1.503e-02

Selectivity matrix (mean |t| for sig-in-row; * = diagonal is row max):

| sig in \ measured | speed | vel_x | vel_y | accel_mag | direction |
|---|---|---|---|---|---|
| speed | **6.43*** | 4.73 | 3.90 | 4.22 | 3.70 |
| vel_x | 3.60 | **7.37*** | 3.84 | 3.98 | 3.60 |
| vel_y | 3.61 | 4.83 | **6.79*** | 3.92 | 6.21 |
| accel_mag | 3.74 | 4.72 | 3.76 | **6.07*** | 3.52 |
| direction | 3.62 | 4.87 | 6.71 | 3.85 | **6.60*** |

**Diagonal-dominant: False**

## PCA (512 sign-split features)   (D=512, threshold p<9.77e-05)

| tau | sig+ | sig- | total% |
|---|---|---|---|
| speed | 51 | 86 | 26.76% |
| vel_x | 96 | 80 | 34.38% |
| vel_y | 64 | 68 | 25.78% |
| accel_mag | 60 | 110 | 33.20% |
| direction | 51 | 60 | 21.68% |

**Significant for >=1 tau: 351/512 (68.55%)** | non-sig mean |C| = 7.746e-03

Selectivity matrix (mean |t| for sig-in-row; * = diagonal is row max):

| sig in \ measured | speed | vel_x | vel_y | accel_mag | direction |
|---|---|---|---|---|---|
| speed | **5.75*** | 4.03 | 2.96 | 3.74 | 3.08 |
| vel_x | 3.13 | **6.53*** | 3.16 | 3.36 | 3.02 |
| vel_y | 2.99 | 3.93 | **5.94*** | 3.36 | 5.36 |
| accel_mag | 3.53 | 3.96 | 3.01 | **5.51*** | 2.83 |
| direction | 3.12 | 4.09 | 5.92 | 3.43 | **6.03*** |

**Diagonal-dominant: True**

## ICA (512 sign-split features)   (D=512, threshold p<9.77e-05)

| tau | sig+ | sig- | total% |
|---|---|---|---|
| speed | 54 | 91 | 28.32% |
| vel_x | 112 | 80 | 37.50% |
| vel_y | 72 | 70 | 27.73% |
| accel_mag | 72 | 96 | 32.81% |
| direction | 58 | 67 | 24.41% |

**Significant for >=1 tau: 352/512 (68.75%)** | non-sig mean |C| = 8.386e-04

Selectivity matrix (mean |t| for sig-in-row; * = diagonal is row max):

| sig in \ measured | speed | vel_x | vel_y | accel_mag | direction |
|---|---|---|---|---|---|
| speed | **5.88*** | 4.39 | 3.52 | 3.52 | 3.40 |
| vel_x | 3.23 | **6.61*** | 3.36 | 3.56 | 3.06 |
| vel_y | 3.65 | 4.52 | **5.84*** | 3.71 | 5.22 |
| accel_mag | 3.43 | 4.33 | 3.31 | **5.58*** | 3.04 |
| direction | 3.62 | 4.22 | 5.71 | 3.79 | **5.75*** |

**Diagonal-dominant: True**

