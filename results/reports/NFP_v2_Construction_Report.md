# NFP v2: the decorrelated stimulus set — why it exists, how it was built, what it found

Self-contained. Assumes basic knowledge of SAEs and video models. Numbers are in
tables; every metric is defined where it first appears. Companion documents:
Temporal_Feature_Evidence_Report.md (the causal evidence program), FINDINGS.md (lab log). Scripts:
`analysis/design_decorrelated_stimulus.py`, `data/nfp_ball_dataset.py`
(`--profile_spec`), `data/run_nfp_v2_dataset.ps1`, `analysis/nfp_v2_analysis.py`.
Results: `expF2_decorr_design.json`, `expF3_nfp_v2.json`, `expF1b_cbar_v2_steering.json`,
`expF4_v2dir_flip.json`, `expE2_direction_nulls.json`.

## 1. The idea behind the construction

### 1.1 What the NFP test is

The NFP test probes a video model with 3,000 rendered ball videos (white ball, gray
floor, 16 frames). Each video has a scripted velocity profile and a start position
drawn uniformly and independently of the profile. For each SAE feature i and video V,
the statistic is the within-video covariance between the feature's activation at the
ball's token and a kinematic variable tau:

  C_i(V) = (1/T) sum_t (psi_i(V,t) - mean_t psi_i)(tau(V,t) - mean_t tau)

with tau one of {speed, vel_x, vel_y, accel_mag, direction}. A one-sample t-test asks
whether C_i has a consistent sign across the 3,000 videos (Bonferroni bar
p < 0.05/6144). The independent uniform start position is what makes the test sound:
a feature that only encodes position cannot produce a consistent covariance.

### 1.2 The flaw in v1

The guarantee above concerns position. It says nothing about couplings BETWEEN the
five kinematic variables inside the stimulus itself. If two variables covary within
videos across the dataset, a feature encoding one inherits covariance with the other,
and the test cannot tell them apart. Measured on the v1 profile distribution (8,000
sampled profiles, uniform weights, tau evaluated at the 8 tubelet time steps):

| pair | mean within-video covariance |
|---|---|
| vel_y - direction | **+0.680** |
| vel_x - vel_y | +0.015 |
| speed - vel_y | +0.009 |
| speed - vel_x | -0.006 |
| vel_x - direction | +0.004 |
| all pairs involving accel_mag, speed - direction | < 0.001 |

For scale: the variables have within-video variances of roughly 0.4-1.0, so +0.680 is
a large coupling. Every other pair is already near zero, not by luck but because the
v1 design randomizes the heading uniformly and pairs profile types (accelerating with
decelerating, step-up with step-down); for example Cov(vel_x, speed) = cos(theta)
Var(speed) per fixed-heading video, and E[cos theta] = 0 under a uniform heading.

The direction - vel_y coupling survives for a parity reason. Direction is the angle
theta; vel_y = speed * sin(theta); vel_x = speed * cos(theta). Over the circle, the
integral of theta * cos(theta) is zero (odd integrand), so direction - vel_x cancels
under heading randomization. The integral of theta * sin(theta) is positive (odd times
odd = even integrand), so direction - vel_y is systematically positive NO MATTER how
headings are randomized. Only reweighting the velocity profiles themselves can cancel
it, using the minority of turning arcs (those crossing the +/-pi wrap) that contribute
negative covariance.

Observed consequences of the flaw before v2 existed: the recovered concept direction
for "direction" was nearly identical to the one for vel_y (cosine 0.96; section 3.2);
the direction row of the selectivity matrix peaked on vel_y; and only 2 of the 85
flagged features were direction-tagged.

### 1.3 The fix

Covariance is bilinear, so the dataset-level coupling is a weighted average over
profiles: sum_j w_j Cov_t(tau_k, tau_a)^(pi_j). Choosing the weights w_j (how many
videos use each profile) makes decorrelation a linear program:

  find w >= 0, sum w = 1, such that sum_j w_j Cov_t(tau_k, tau_a)^(pi_j) = 0 for all
  pairs k != a, and the weighted variance of every variable stays above a floor.

There is a hard limit (Chebyshev's sum inequality): a confound that is a monotone
function of the target has single-signed covariance in every profile and cannot be
decorrelated by any stimulus design. Checked empirically: for every (target, confound)
pair in our five variables, the per-profile covariances carry both signs, so no pair
in our set is impossible.

## 2. Implementation detail

### 2.1 Solving the design

- Profile pool: M = 4,000 profiles sampled from the unmodified v1 generator (family A:
  constant, linear_accel, linear_decel, sinusoidal, slow_fast_slow, fast_slow_fast,
  step_accel, step_decel; family B: gradual_turn, sharp_turn, back_and_forth; speeds
  and headings uniform). tau computed at the 8 tubelet steps (frame 2*step), matching
  the NFP pipeline.
- The joint LP zeroes all 10 pairwise couplings simultaneously, with a variance floor
  of 0.5x the pool mean for every variable. Solver: scipy linprog (highs) with a
  spread cap w_j <= 3/M, which forces the solution to use many profiles (effective
  sample size 1/sum(w^2) = 1,335 of 4,000; effective sample size means the weighted
  set behaves statistically like that many equally-weighted profiles).

Joint solution quality:

| quantity | value |
|---|---|
| max residual coupling (weights) | 3.4e-9 |
| weighted Var: speed | 0.549 (floor 0.199) |
| weighted Var: vel_x | 0.563 (floor 0.482) |
| weighted Var: vel_y | 0.503 (floor 0.503) |
| weighted Var: accel_mag | 0.0012 (floor 0.0004) |
| weighted Var: direction | 0.501 (floor 0.501) |

- Exact allocation to N = 3,000 videos by largest-remainder rounding of N * w_j, then
  a deterministic shuffle. Videos per profile type:

| profile type | videos |
|---|---|
| B: gradual_turn | 462 |
| B: sharp_turn | 333 |
| A: fast_slow_fast | 322 |
| A: step_accel | 292 |
| A: linear_decel | 286 |
| A: step_decel | 248 |
| A: sinusoidal | 247 |
| A: slow_fast_slow | 245 |
| A: linear_accel | 232 |
| A: constant | 191 |
| B: back_and_forth | 142 |

The per-video spec (profile family, type, speed, heading, turn angle) is saved to
`data/nfp_v2_profile_spec.json`.

### 2.2 Generation

- `nfp_ball_dataset.py` gained `--profile_spec`: video idx uses the spec's idx-th
  profile, reconstructed deterministically from its saved parameters. The start
  position is still drawn from the same per-video RNG as v1 (seeded by [seed, idx]),
  so the position-independence assumptions of the NFP proof hold unchanged, and v2's
  start positions are bit-identical to v1's. v2 differs from v1 in exactly one thing:
  the velocity profiles.
- Rendered with the same Kubric Docker image as v1, in three resume-safe shards, to
  `data/output/nfp_v2/` (3,000 videos).

### 2.3 Verification of the rendered set

Recomputed from the rendered metadata (not the design weights), all 10 couplings:

| pair | v1 | v2 rendered |
|---|---|---|
| vel_y - direction | +0.680 | **+0.00044** |
| vel_x - vel_y | +0.015 | -0.0012 |
| speed - vel_x | -0.006 | -0.0022 |
| speed - vel_y | +0.009 | -0.0011 |
| all remaining pairs | < 0.001 | < 0.0005 |

The residuals come from integer rounding of the weights and are negligible. The
largest v1 coupling was cut 1,500x.

## 3. What the new set found

All measurements below use the SAME SAE (trained on SSv2 activations; nothing about
the model or dictionary changed) and the same NFP statistic and bar. Only the probe
stimulus changed.

### 3.1 The flagged features

| quantity | v1 | v2 |
|---|---|---|
| features flagged | 85 (1.38%) | 109 (1.77%) |
| overlap | - | 70 of the v1 85 re-flagged |
| speed: significant / dominant | 40 / 27 | 66 / 54 |
| vel_x: significant / dominant | 40 / 33 | 39 / 28 |
| vel_y: significant / dominant | 24 / 14 | 33 / 14 |
| accel_mag: significant / dominant | 13 / 9 | 19 / 10 |
| direction: significant / dominant | 15 / 2 | **4 / 3** |

("Significant" = passes the bar for that variable; "dominant" = that variable has the
feature's largest |t|.) Two readings. First, the core is stable (70/85) and the total
grew because the design carries more speed variance (0.55 vs 0.40), which raises speed
covariances. Second, the direction column: v1's 15 direction-significant features
collapse to 4. Most of v1's direction significance was the vel_y coupling leaking
through the test. The v2 direction-dominant features are three and all NEW indices:
feat00115, feat01217, feat03747 — genuine direction detectors that the confounded
stimulus masked. The v1 direction tag was mostly artifact; the v2 tag is honest and
small.

### 3.2 The recovered concept directions (c_bar)

c_bar_tau = the mean within-video covariance vector between the 768-d ball-token
activation and tau; the direction in representation space that best detects tau under
the stimulus (closed form, no training). Cosines between recovered directions:

| pair of c_bar directions | v1 | v2 |
|---|---|---|
| direction , vel_y | **+0.96** | **-0.17** |
| speed , accel_mag | +0.37 | +0.34 |
| speed , direction | -0.06 | -0.22 |
| vel_y , accel_mag | +0.05 | +0.23 |
| all others | <= 0.11 | <= 0.13 |

Under v1, the direction and vel_y directions were the same vector; the probe could not
distinguish the two concepts. Under v2 all five directions are mutually distinct. This
is the cleanest single demonstration that the stimulus coupling, not the model, caused
the entanglement.

### 3.3 The selectivity matrix, reinterpreted

Selectivity = for features flagged on variable A, their mean |C| response on variable
B, with each variable z-scored globally so columns are comparable (a response of 0.03
means the average flagged feature moves 0.03 z-units of covariance). v2 values, rows =
flagged-for, columns = response:

| flagged for | speed | vel_x | vel_y | accel | direction | diagonal-dominant? |
|---|---|---|---|---|---|---|
| speed | **0.030** | 0.013 | 0.011 | 0.003 | 0.003 | yes |
| vel_x | 0.020 | **0.030** | 0.007 | 0.003 | 0.004 | yes |
| vel_y | 0.029 | 0.014 | 0.019 | 0.002 | 0.004 | no (speed) |
| accel_mag | 0.028 | 0.014 | 0.008 | 0.011 | 0.005 | no (speed) |
| direction | 0.024 | 0.012 | 0.011 | 0.002 | 0.011 | no (speed) |

Three rows peak on speed. This can no longer be blamed on the stimulus (its couplings
are zero), so it is a property of the features: a ReLU feature that detects, say,
upward motion fires only on its preferred side, and within those videos its activation
tracks speed, so Cov(feature, speed) > 0 even when Cov(vel_y, speed) = 0 in the
stimulus. Rectified direction-of-motion detectors are speed detectors on their
preferred side, and no stimulus design can prevent that (the same Chebyshev-flavor
argument as the monotone-confound limit). Consequence: on a decorrelated stimulus the
selectivity matrix measures the features' true nonlinear response profile; on v1 it
measured a mixture of that and the stimulus confounds.

### 3.4 The directions and features in steering

Full protocols and definitions in Temporal_Feature_Evidence_Report.md section 2.1. Summary of what the
v2 objects do when steered (add delta * unit direction, or clamp the feature, on real
SSv2 videos of the matched class pairs):

| object | pair | result | control |
|---|---|---|---|
| c_bar[vel_x] (v2) | camera left/right | shift 4.08, flips 42%, top-1 25% | 30-direction null: max shift 2.52, max flips 12%, p = 0/30 |
| c_bar[vel_x] (v1 version) | camera left/right | shift 3.80, flips 29%, top-1 12% | same nulls |
| c_bar[vel_y] (v2) | camera up/down | shift 3.24, flips 25% | null max 2.90, p = 0/30 |
| c_bar[direction] (v2) | camera left/right | shift 1.48, flips 0% | within null range |
| feat01217 (new, direction-dominant) | pushing left/right | shift -3.04, flips 4% -> 25% | best v1 feature: shift -1.66, flips <= 8% |
| feat00115, feat03747 (new) | all pairs | shifts <= 1.1, flips <= 12% | weak |
| head-diff (ceiling, from classifier rows) | every pair | flips 100% | - |

Readings. Decorrelation sharpened the one contaminated direction: c_bar[vel_x] gained
about 1.4x on flips and 2x on strict top-1. feat01217 is the first probe-flagged
feature to move the object-level pushing pair at all, with a caveat: its steering
endpoint (computed from the weights) is the class "Pushing [something] with
[something]" at p = 0.72, and it moves the up/down pair equally, so the movement is
partly attraction toward pushing classes rather than a clean left/right knob. The
disentangled c_bar[direction] still steers nothing, and its cosine to the classifier's
own direction readouts stays <= 0.27: the direction the model READS for left/right
judgments is carried by real-scene features that ball videos never activate. A perfect
ball stimulus sharpens what the probe can see; it does not extend what the probe can
reach.

## 4. Summary

1. v1 had one large stimulus coupling (direction - vel_y = +0.68), unfixable by
   heading randomization for a parity reason, and it contaminated direction tags,
   the direction concept vector, and the selectivity matrix.
2. A linear program over velocity-profile weights zeroes all 10 couplings at once;
   the rendered v2 set verifies at <= 0.0022 per pair with positions identical to v1.
3. On v2, the same SAE and test give: a stable 70-feature core plus honest tags
   (direction: 15 significant -> 4, three new genuine direction features); fully
   disentangled concept directions (direction - vel_y cosine 0.96 -> -0.17); and a
   selectivity matrix that now measures feature nonlinearity (rectified velocity
   detectors are speed detectors on their preferred side) instead of stimulus
   confounds.
4. In steering, the decorrelated c_bar[vel_x] is sharper (42% flips, no null direction
   among 30 comes close), and the probe's domain limit is now cleanly separated from
   its statistical quality: better statistics improved recovery of what ball videos
   can show, and left the model's object-direction machinery, which lives in
   ball-silent features, exactly as unreachable as before.
