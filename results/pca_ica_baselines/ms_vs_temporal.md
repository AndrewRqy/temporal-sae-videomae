# Are the SAE's temporal features its high-MS features? (cross-search)

**Question (issue #3, line 4):** for the SAE, do the features that the NFP test flags as
**temporal** also have **high monosemanticity (MS)** scores?

**Method.** For the *same* local SAE, join the per-feature MS score (`all_neurons_scores.pth`,
one MS value per feature, see `summary.md` §2a) with the NFP temporal-significance mask
(`sae_nfp.pt`: a feature is "temporal" if its within-video covariance t-test passes
`p < 0.05/6144` for ≥1 τ, see §2b). Then compare the MS distribution of the temporal features
against all features. Reproduce: `python analysis/ms_vs_temporal.py`.

**Result.**

| Group | n | MS mean | MS median | MS max | # > 0.6 | # > 0.7 |
|---|---|---|---|---|---|---|
| All (live) features | 6088 | 0.468 | 0.470 | **0.782** | 118 | 8 |
| **NFP-temporal** features | 85 | **0.441** | 0.455 | **0.542** | 0 | 0 |
| NFP-nonsignificant | 6003 | 0.468 | 0.470 | 0.782 | 118 | 8 |

- The temporal features' MS is **slightly below average** (0.441 vs 0.468), not high.
- Their MS **maxes out at 0.54** — *none* exceed 0.60, whereas the SAE overall has 8 features
  above 0.70 (peak 0.78). So the SAE's celebrated **high-MS tail contains no temporal features**.
- The temporal features sit at the **~36th percentile** of the MS distribution (below the median),
  and **0 of the top-100 MS features are temporal**.
- Per-τ, every temporal subset is the same story (MS mean ≈ 0.42–0.45, max ≈ 0.50–0.54):

| τ | # temporal | MS mean | MS max |
|---|---|---|---|
| speed | 40 | 0.438 | 0.532 |
| vel_x | 40 | 0.437 | 0.542 |
| vel_y | 24 | 0.445 | 0.512 |
| accel_mag | 13 | 0.425 | 0.504 |
| direction | 15 | 0.445 | 0.512 |

**Answer: No.** The SAE's temporal features are *not* high-MS — if anything they are marginally
below average, and the high-MS tail is entirely non-temporal (presumably static/appearance
features).

**Why this is expected, and what it implies (ties to issue #3, line 5).** The MS score measures
whether a feature fires on a **visually similar** set of clips, using **DINOv2 image embeddings**
as the similarity (`summary.md` §2a). But a genuine *temporal* feature — e.g. "the object is
moving fast" — fires across clips that look **visually different** (a fast car, a fast hand, a fast
ball). Image similarity is therefore low for those clips, so the feature scores **low MS even
though it is a perfectly good temporal feature**. In other words, the current MS metric
**structurally undercredits temporal features**: it is the wrong yardstick for them. This is direct
evidence for the line-5 proposal to **redesign MS for temporal concepts** (e.g. weight by a
motion/temporal similarity instead of DINOv2 image similarity, or — as already done on the
synthetic control — score against the ground-truth temporal subspace `W_τ` rather than image
embeddings).
