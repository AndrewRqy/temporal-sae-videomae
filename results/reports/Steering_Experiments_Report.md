# Steering temporal SAE features in VideoMAE: methods and results

Self-contained. Assumes basic knowledge of representation spaces, nothing more.
Companion documents: FINDINGS.md (chronological lab log), PROPOSAL.md (original plan).

## 1. Setup and the steering operation

### 1.1 The system

VideoMAE (`MCG-NJU/videomae-base-finetuned-ssv2`) watches a 16-frame clip and outputs a
probability over 174 action classes from the Something-Something-v2 (SSv2) dataset. It
processes the clip as 1568 tokens (8 time slices x 196 spatial patches), each a
768-dimensional vector, through 12 transformer layers. The prediction comes from
mean-pooling the final token vectors, applying LayerNorm, and applying a linear head.

Our sparse autoencoder (SAE) re-expresses each layer-11 token vector as a sparse
combination of 6,144 dictionary features: encode(x) = ReLU(W_enc(x - b)), decode(f) =
W_dec f + b. Each feature k has a decoder column mu_k, the direction added to the
representation when the feature is active.

The NFP test previously flagged 85 of the 6,144 features as temporal, each tagged with
the kinematic variable it tracks best (speed, vel_x, vel_y, accel_mag, direction). This
report asks whether those features causally influence the model's predictions.

### 1.2 The steering operation

To set feature k to strength s on one token vector x:

1. f = encode(x)
2. e = x - decode(f)        (reconstruction error, saved)
3. f_k <- s
4. x_out = decode(f) + e    (error added back)

This reduces to x_out = x + mu_k (s - f_k): the vector moves along feature k's decoder
direction and nothing else changes. Adding e back avoids a reconstruction penalty.
Because decode is linear in f, negative s is valid, so one feature is a two-way knob.
We apply this to all 1568 tokens, finish the forward pass, and read the class
probabilities. Baseline sanity: top-1 accuracy 68.4% on a 256-clip sample of real
SSv2 validation videos, which all experiments below use as inputs.

### 1.3 Key parameters

Concepts per feature come from its top-25 activating clips out of a 384-clip pool.
Clamp strengths: s in {-150 ... +150} depending on experiment. 12-24 videos per class
where classes are the unit. Random-feature nulls throughout. Seeds fixed at 0. All
scripts in `analysis/steer_*.py`; all outputs in `local_runs/steering/`.

## 2. Experiment A: steering toward a feature's own concept

### 2.1 Population test

Define each feature's concept as the SSv2 classes of its own top-activating clips.
Steer the feature and measure whether probability mass moves toward those classes,
z-scored against 500 label permutations, then compared to random features.

Result: the 85 NFP features steer toward their own concepts more than random features.
Mann-Whitney p = 2.5e-4 at s=100 and 2.1e-3 at s=40. The per-feature effect correlates
r = 0.97 between the two clamp strengths, so it is a stable property of specific
features, not a large-clamp artifact.

A caution that shaped everything after: hand-built readout axes (a fast/slow class set,
a motion keyword set) never separate NFP features from random ones. A hard clamp moves
the output for any feature. Only readouts tied to each feature's own target separate
signal from null.

### 2.2 The bidirectional speed-axis test (4-cell)

A feature can pass the concept test by pulling every input toward one fixed output (see
Section 6 on attractors). To separate genuine motion control from that, we built an
objective motion axis (mean optical-flow magnitude per class, z-scored across the 174
classes) and read out E[speed] = sum_c P(c) speed_z(c). For each feature: pick a fast
and a slow class from its own concept set, hold real videos of those classes fixed, and
steer up and down from both starting points. A genuine motion knob must raise E[speed]
when steered up from both starts and lower it when steered down.

Result: 9 of 85 features pass (both slopes positive and slope-z >= 2 against a random
null): feat01321, feat02509, feat03968, feat00950, feat04665, feat01699, feat03672,
feat00214, feat03059. Most other features show the converging pattern instead: slope
negative from fast starts, positive from slow starts. That is convergence to a fixed
output, not motion control.

### 2.3 Generalization

The 9 were then steered on a 12-class panel spanning the motion spectrum, classes they
were never tuned on (camera-pan classes excluded to avoid the optical-flow confound).
All 9 raise predicted motion on 92-100% of panel classes; random features do so on
61% +/- 19%. Three (feat01321, feat02509, feat00950) hit 100%. The 9 are content-general
motion knobs, not artifacts of their own anchor classes.

### 2.4 Dose-response demos

On motionless inputs (one frame repeated 16x), cranking one feature makes the classifier
predict that feature's specific motion, monotonically in s: feat02312 raises
P(Spinning that quickly stops) from 0.007 to 0.30; feat03714 raises P(Pushing so it
slightly moves) to 0.34; feat04280 raises P(Lifting) to 0.19. The same curves hold on
real inputs at slightly lower ceilings.

### 2.5 Supervised ceiling

A difference-of-means direction (mean layer-11 residual over fast-class clips minus
slow-class clips) steers the fast/slow axis about 7x harder than any single feature.
Single SAE features are not the most powerful steering directions. Their value is that
the dictionary provides 85 pre-labeled directions with no supervision.

## 3. Experiment B: flipping direction pairs

SSv2 contains class pairs identical in content that differ only in motion direction:
Pushing/Pulling left-to-right vs right-to-left, Moving up vs down. The model separates
all pairs at 96-100% unsteered. Can one feature flip the prediction across a pair?

### 3.1 NFP features

Single NFP vel features fail on the horizontal pairs (flips <= 8%) and partially work
on up/down (feat04280, feat04541: 42% pair-restricted flips from a 4% base). The NFP
t-sign does not predict which way a feature steers (11/24 agreement, chance).

### 3.2 The ceiling and the explanation

A per-pair difference-of-means vector flips every pair at 100%, both pair-restricted
and strict top-1, at small strength. So direction is linearly encoded at layer 11 and
fully steerable; the failure is in the features, not the layer. Decomposing each pair's
direction axis onto the 6,144 decoder columns shows the axis is concentrated in a few
features (single-column cosine up to 0.77), but none of the top-10 carriers for any
pair is among the 85 NFP features. Most have t = NaN in the NFP results: they never
activated on the synthetic ball videos. The model's real-scene direction features are
out-of-distribution for the ball probe, so NFP could not flag them.

### 3.3 Flipping with the axis features

Steering those features flips the pairs: feat05672 flips Pushing left-to-right to
right-to-left at 100% strict top-1; feat01652 at 88% (pulling); feat01387 and feat05258
at ~80% (up/down). The correct sign calibrator is the feature's baseline activation
difference between the two classes (12/12 correct); the NFP sign remains useless here.
Caveat: these features were located using class labels (via the supervised axis), so
this result demonstrates a property of the SAE dictionary and the classifier, not a
validation of NFP. It is reported as such.

## 4. The systematic pair screen and its controls

### 4.1 Screen

All 85 NFP features x 10 temporal pairs (pushing, pulling, camera pan left/right,
moving up/down, camera up/down, three depth pairs, falling like a rock vs feather,
spinning continues vs stops), steered at +/-150 on both sides of every pair. No feature
selection, no classifier-derived directions.

Results:
- 13 of 85 features reach >= 50% flips on at least one pair. Highlights: feat00250
  flips spinning-continues vs quickly-stops at 100% (its concept from Experiment A was
  "Spinning so it continues spinning"); feat04665 flips camera up/down at 0.79;
  feat01321 and feat00214 flip camera left/right at 0.75 and 0.71; three speed features
  flip falling-rock vs feather at 0.54-0.67.
- Effects organize by NFP tag: axis-matched pairs get mean |log-odds shift| 0.64 vs
  0.40 for mismatched pairs (Mann-Whitney p = 3.9e-5). Only vel-tagged features flip
  the camera-pan pairs at all.
- Built-in negative control: the three depth pairs match no NFP variable, and NFP
  features flip them at 0.039 vs 0.094 elsewhere. The features are not generic
  disruptors.
- Five of the proven-9 knobs are also top pair-flippers. Same features keep surfacing
  under independent protocols.
- Honest negative: object-level push/pull direction stays unflippable by NFP features
  (max 21%), consistent with Section 3.2.

### 4.2 Random static features (sufficiency control)

30 random features with finite NFP t and max |t| < 2 (active on ball videos,
measurably non-temporal) went through the identical screen. They shift pair log-odds
somewhat (0.335 vs 0.447 for temporal, Mann-Whitney p = 3.9e-3) but never flip:
0 of 30 reach 50% on any pair, vs 12 of 85 temporal (Fisher p = 0.021). Flipping is
exclusive to the temporal set.

### 4.3 Ablation (necessity control)

Clamping feature sets to zero (all 85, the 12 flippers, matched random sets) leaves
pair accuracy exactly unchanged (0.967) on every pair. The intervention is real: it
removes 17% of the layer-11 token norm and erodes camera-pair margins by 0.5-1.0
log-odds while random sets erode nothing. Accuracy does not move because baseline
margins are 4.5-5.4 log-odds and no video sits near the boundary.

Repeating on the hardest videos (the 12 per side nearest the decision boundary,
margins down to 0.8): decision flips remain near zero for every set. Margin damage
orders as predicted: the 30 axis-carrier features damage most (-0.144, significant vs
matched random sets, Wilcoxon p = 0.016 and 0.007 on two seeds), the temporal sets
about -0.10 (a trend), random sets ~0. Absolute effects stay small, about 6% of margin.

Conclusion: the NFP features are sufficient to steer temporal-pair classification and
static features are not, but no 30-85 feature subset is necessary for it. The model
encodes temporal properties redundantly. Levers, not wires.

## 5. Why steering strength and detection strength are different things

Input score (NFP detection strength, max |t|) against three output scores (own-concept
alignment z, 4-cell slope z, pair-screen power) across the 85 features:
r = +0.07, -0.00, +0.08, all p > 0.26. Top-quartile co-occurrence is at chance.
Detection strength carries no information about steering power, in either direction.
The strongest detector in the dictionary (feat05179, max |t| = 13.2) is inert on all
three output measures, and none of the 9 knobs is a top detector.
Figure: `figures/input_output_scatter.png`.

This is the classification-setting form of the input-vs-output feature distinction in
the LLM steering literature (Arad et al. 2025; Chalnev et al. 2024). Steering
candidates should be selected by output-side tests, not by detection statistics.

## 6. Attractors: the correct frame for most of what steering does

Steering shifts class c's logit by (s - f_k) alpha_c, where alpha_c is the dot product
of class c's readout row with mu_k. These alpha_c are fixed numbers, independent of the
input. At large s every input converges to the same output distribution. Because the
head applies LayerNorm before the linear readout and LayerNorm is scale-invariant, the
limit is exact: p_attractor(k, sign) = softmax(W LN(sign mu_k) + b), computable from
the weights with no forward passes. The +s and -s attractors are different
distributions, not mirrors.

Computed for all features, this theory predicts the measurements:
- The sign of [attractor E-speed(+s) minus E-speed(-s)] matches the measured 4-cell
  slope sign for 63 of 77 features (82%), from weights alone.
- The 9 knobs have graded two-sign speed contrast (|dE-speed| = 2.36) and moderate
  attractor sharpness (0.64). The direction-flip features are sharp attractors
  (0.78) aimed at direction classes: feat05672's +s attractor is "Pushing from right
  to left" at p = 0.94.

Two framing consequences. First, attractor vs knob is a property of the pair (decoder
column, classification head) and of the axis you evaluate on, not of the feature
alone. Second, attractor behavior is not deceptive per se: the 100% direction flips in
Section 3.3 are attractor behavior pointed at the desired class, and feat02312's
dose-response demo is partly attraction toward its own concept class. The earlier
phrasing "regression to an attractor masquerades as steering" is replaced by: the
attractor pattern is not evidence of a motion-specific causal role with respect to the
chosen axis; it reflects alignment between the decoder column and the head.

## 7. Summary of claims

1. NFP temporal features causally steer the classifier toward their own concepts at the
   population level (p = 2.5e-4), robustly across clamp strengths (r = 0.97).
2. Nine features are content-general motion knobs: direction-consistent under the
   bidirectional test and effective on 92-100% of unseen classes.
3. In an unselected screen over 10 temporal class pairs, 13 of 85 features flip at
   least one pair at >= 50%, effects organize by NFP tag (p = 3.9e-5), depth pairs act
   as a passing internal negative control, and random static features never flip
   (Fisher p = 0.021).
4. The features are sufficient but not necessary: ablating all 85 leaves pair accuracy
   unchanged. Temporal information is redundantly encoded.
5. Detection strength and steering power are independent (all |r| < 0.09). Selecting
   steering features requires output-side tests. The attractor spectra, computed from
   weights alone, predict steering direction at 82% and separate knobs from sharp
   attractors.
6. Limits: single features steer ~7x weaker than a supervised direction; horizontal
   object-motion direction lives in features the ball probe cannot see (they are
   silent on synthetic balls); the 100% flip demo uses supervised feature selection
   and is a claim about the dictionary, not about NFP.
