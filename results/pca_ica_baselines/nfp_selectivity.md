# NFP test — selectivity (diagonal) scores: raw layer / PCA / ICA

Within-video covariance + one-sample t-test over 3000 ball videos. A feature is
significant for tau k if p < 0.05/D (Bonferroni). Selectivity matrix: for features
significant in the ROW tau, the mean |t-stat| across each COLUMN tau; the diagonal
(bold *) should be the row max if features are concept-selective. SAE reference
(cluster): 75/6144 (1.2%) significant, diagonal-dominant.

## Raw layer (768 dims, identity)   (D=768, threshold p<6.51e-05)

| tau | sig+ | sig- | total% |
|---|---|---|---|
| speed | 145 | 147 | 38.02% |
| vel_x | 200 | 201 | 52.21% |
| vel_y | 164 | 147 | 40.49% |
| accel_mag | 213 | 168 | 49.61% |
| direction | 148 | 143 | 37.89% |

**Significant for >=1 tau: 698/768 (90.89%)** | non-sig mean |C| = 1.503e-02

Selectivity matrix (mean |t| for sig-in-row across columns; * = diagonal is row max):

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

Selectivity matrix (mean |t| for sig-in-row across columns; * = diagonal is row max):

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

Selectivity matrix (mean |t| for sig-in-row across columns; * = diagonal is row max):

| sig in \ measured | speed | vel_x | vel_y | accel_mag | direction |
|---|---|---|---|---|---|
| speed | **5.88*** | 4.39 | 3.52 | 3.52 | 3.40 |
| vel_x | 3.23 | **6.61*** | 3.36 | 3.56 | 3.06 |
| vel_y | 3.65 | 4.52 | **5.84*** | 3.71 | 5.22 |
| accel_mag | 3.43 | 4.33 | 3.31 | **5.58*** | 3.04 |
| direction | 3.62 | 4.22 | 5.71 | 3.79 | **5.75*** |

**Diagonal-dominant: True**

