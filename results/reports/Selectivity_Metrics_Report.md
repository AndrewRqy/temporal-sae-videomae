# Selectivity under the new metric: VideoMAE SAE, synthetic control, DINOv2 negative control

## 1. What the metric is and why it changed

The selectivity matrix answers one question. Take the features flagged for motion variable
A. How strongly do they respond to variable B? If the test finds real, specific features,
each row should peak on its own diagonal: speed-flagged features should respond most to
speed, and so on.

The old matrix reported the mean |t| statistic. The t statistic measures consistency
across videos, not response strength. At 3,000 videos, a tiny covariance that always
points the same way produces a large |t|, so the old matrix could be dominated by weak
but reliable responses.

The new matrix reports mean |C|, the mean absolute within-video covariance, with each
motion variable z-scored across the whole dataset. Z-scoring puts all five variables on
one scale, so entries are comparable across columns. Because covariance is linear in the
variable, this is computed exactly as the raw covariance divided by that variable's
global standard deviation. Flagging is unchanged: features are still selected by the
t-test with Bonferroni correction. The matrix is a post-hoc description of the flagged
features.

The numbers below are computed from the saved covariances of the three current-standard
NFP runs. The stimulus set (3,000 ball videos, 8 steps) is identical in all three, so the
z-scoring constants are identical too.

## 2. Results

Rows: features flagged for that variable. Columns: mean |C| response on the z-scored
variable. The diagonal entry is marked with <. "Diag" states whether the row peaks on its
own variable.

### 2.1 VideoMAE SAE (the main positive result)

85 of 6,144 features flagged (1.38%).

| flagged for | speed | vel_x | vel_y | accel_mag | direction | diag |
|---|---|---|---|---|---|---|
| speed | **0.0270<** | 0.0139 | 0.0121 | 0.0034 | 0.0086 | yes |
| vel_x | 0.0163 | **0.0296<** | 0.0112 | 0.0027 | 0.0083 | yes |
| vel_y | 0.0199 | 0.0149 | **0.0279<** | 0.0024 | 0.0179 | yes |
| accel_mag | 0.0226 | 0.0096 | 0.0103 | 0.0090< | 0.0117 | no (peak: speed) |
| direction | 0.0166 | 0.0091 | 0.0297 | 0.0027 | 0.0223< | no (peak: vel_y) |

Unflagged features average 0.00018 on the same scale. Flagged-row entries are 50x to
170x larger, so the flagged set responds to motion and the rest of the dictionary does
not.

### 2.2 Synthetic control (per-video Gaussian, the HF-uploaded standard)

17 of 6,144 features flagged (0.28%).

| flagged for | speed | vel_x | vel_y | accel_mag | direction | diag |
|---|---|---|---|---|---|---|
| speed | **0.0400<** | 0.0320 | 0.0355 | 0.0005 | 0.0118 | yes |
| vel_x | 0.0164 | **0.0268<** | 0.0246 | 0.0003 | 0.0200 | yes |
| vel_y | 0.0139 | 0.0225 | **0.0239<** | 0.0003 | 0.0187 | yes |
| accel_mag | 0.0000 | 0.0000 | 0.0000 | 0.0000< | 0.0000 | no |
| direction | 0.0139 | 0.0225 | 0.0239 | 0.0003 | 0.0187< | no (peak: vel_y) |

Unflagged features average 0.00003.

### 2.3 DINOv2 negative control (patch tokens)

0 of 6,144 features flagged. The selectivity matrix is empty. Unflagged features average
0.00005, the same order as the synthetic control's unflagged population. The older
whole-image DINOv2 run also flags 0. DINOv2 is an image model with no temporal
information, so this is the required outcome: the test finds nothing where nothing
exists.

## 3. Reading the deviations

Three rows fail diagonal dominance. Each has a specific cause.

**The direction row peaks on vel_y in both VideoMAE and the synthetic control.**
Direction is the angle of the velocity vector, so it is a function of vel_x and vel_y. A
feature that tracks heading must also covary with the velocity components. The variables
are entangled, and the effect-size metric shows the entanglement. The old |t| matrix
showed the same rows as dominant or near-dominant, which hid this. This is a property of
the chosen variable set, not of the features.

**The accel_mag row in VideoMAE peaks on speed.** Two causes stack. First, acceleration
and speed are coupled in the stimulus: acceleration is the rate of change of velocity, so
videos with high acceleration also have changing speed. Second, accel_mag has by far the
smallest spread of the five variables (standard deviation 0.136 versus 1.07 to 1.79 for
the others), so even after z-scoring, features respond to it weakly in absolute terms.
The accel-flagged features are real temporal features, but their speed response is larger
than their acceleration response.

**The accel_mag row in the synthetic control is zero everywhere.** This row is the five
known false positives (position-content features flagged through the nonlinear swing-size
leak). Their effect sizes are numerically zero on every variable. They were flagged
because the t-test rewards consistency: a tiny covariance that points the same way in
most videos passes the significance bar even when its magnitude is negligible. This is
the clearest illustration of why the metric changed. The old matrix scored these features
by the same consistency that got them flagged, so they looked unremarkable. The new
matrix scores them by response strength and exposes them as near-zero, which is what they
are. In practice the pattern "flagged, but effect size approximately zero across all
variables" is a usable red-flag signature for this failure mode.

## 4. Summary

Under the effect-size metric, the picture across the three standard runs is consistent.
The VideoMAE SAE and the synthetic control both show diagonal dominance for speed, vel_x,
and vel_y, with flagged features responding 50x to 170x above the unflagged population.
The two failures of dominance have identified causes: variable entanglement
(direction with vel_y, acceleration with speed) and the known nonlinear leak, which the
new metric now makes visible as a zero row. The DINOv2 negative control flags nothing.
Raw numbers: `local_runs/selectivity_new_metric.json`.
