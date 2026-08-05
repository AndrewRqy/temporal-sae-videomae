# The evidence that the NFP-identified features and directions carry temporal meaning
# for the model's classification

Self-contained. Assumes basic knowledge of SAEs and video models. Each experiment is
described by design, parameters, intended effect, observations, and ablations, and
every number is defined where it first appears: how it is calculated and what it means.
Experiments are grouped by which NFP dataset produced the objects being tested: the
original (v1) ball dataset and the decorrelated (v2) ball dataset. Companion documents:
Steering_Experiments_Report.md (steering methods and the clamp experiments), the causal-experiments report (causal
experiments, shorter form), FINDINGS.md (chronological lab log, 39 entries). All
scripts are in `analysis/`, all result files in `local_runs/steering/`.

## 0. Shared setup

### 0.1 The system

VideoMAE (`MCG-NJU/videomae-base-finetuned-ssv2`) classifies 16-frame clips into 174
SSv2 action classes. It represents a clip as 1568 tokens (8 time slices x 196 spatial
patches), each a 768-dimensional vector, processed through 12 transformer layers. The
prediction comes from mean-pooling the final token vectors, applying LayerNorm, and a
linear head. Baseline sanity: 68.4% top-1 on a 256-clip validation sample.

A sparse autoencoder (SAE) re-expresses each layer-11 token vector x as a sparse code
over 6,144 features: encode(x) = ReLU(W_enc(x - b)), decode(f) = W_dec f + b. Feature k
has decoder column mu_k (unit norm), the direction added to the representation when the
feature activates.

The NFP test flags features whose within-video covariance with a kinematic variable
tau (speed, vel_x, vel_y, accel_mag, direction) is consistent across 3,000 synthetic
ball videos. The statistic: for each feature i and video V, C_i(V) =
(1/T) sum_t (psi_i(V,t) - mean_t psi_i)(tau(V,t) - mean_t tau), where psi_i is the
feature's activation at the ball's token. A one-sample t-test asks whether C_i has a
consistent sign across the 3,000 videos; the bar is Bonferroni-corrected, p < 0.05/6144
= 8.14e-6. On the v1 dataset this flags 85 features; each is tagged with the variable
of its largest |t| (its dominant tag): 27 speed, 33 vel_x, 14 vel_y, 9 accel_mag,
2 direction.

### 0.2 The interventions

All interventions edit the layer-11 representation and re-add the SAE reconstruction
error e = x - decode(encode(x)), so only the chosen features change and everything the
SAE cannot represent is preserved. Four types:

1. **Clamp**: set feature k's activation to a fixed value s on all 1568 tokens.
   Equivalent to x -> x + mu_k (s - f_k). s can be negative (decode is linear), so one
   feature is a two-way knob.
2. **Patch**: replace the chosen features' token-level activations with values from
   another forward pass (a different video, or the same video before a corruption).
   No value leaves the natural activation range.
3. **Amplify**: patch with the features' own activations scaled by alpha. Inactive
   features stay inactive; the spatial pattern is preserved.
4. **Erase span**: build the orthogonal projector P onto the span of the chosen
   decoder columns (QR decomposition, P = QQ^T) and replace every token x with x - Px.
   This removes everything the model could read along those directions, active or not.
5. **Add direction**: add delta times a unit 768-d vector to every token (for testing
   directions rather than features).

### 0.3 The control sets

Every effect is compared against feature sets or directions of the same size:

- **random**: drawn from the 6,059 non-flagged features. Three independent draws were
  used across the program (seeds 7, 101, 202).
- **static**: non-flagged features with finite NFP t-statistics and max |t| < 2 across
  all five variables. These were active on the ball videos but measurably carried no
  temporal signal, making them a stricter control than arbitrary random features.
- **activation-matched**: for each NFP feature, the non-flagged feature closest in mean
  activation over a 4,300-video sample (greedy nearest neighbor, no replacement).
  Reason: random features average 0.038 mean activation vs 0.083 for NFP features
  (2.2x less active), so any intervention whose strength scales with activation would
  be unfairly weak under a random control. The matched pool aligns to 0.0825 vs 0.0826.
- **all-6144**: every feature. The ceiling; what the intervention achieves when given
  the entire code.
- For direction-level claims: **20 random unit directions** (weak null; a uniform
  random direction in 768-d is nearly orthogonal to everything the model uses) and
  **10 manifold-matched directions** (random unit combinations of the top-50 principal
  components of the ball-token activations; these look like real activity and produce
  about 2x larger null effects, making them the honest "any direction" control).

### 0.4 The main readout: pair log-odds

Most experiments read out on temporal class pairs: SSv2 classes identical in content
that differ in one temporal property (pushing/pulling left-to-right vs right-to-left,
moving/turning-camera up vs down, falling like a rock vs like a feather, spinning that
continues vs quickly stops). The model separates all pairs at 0.96-1.00 accuracy
unsteered, so the readout is valid.

**Pair log-odds**: LO = log P(class A) - log P(class B), where P comes from the
model's softmax over 174 classes, computed per video and averaged. Meaning: LO = 0 is
indifference; LO = 1 means A is favored e ~ 2.7:1 within the pair; typical unsteered
margins are 4.5-5.4 (90:1 to 220:1 odds). A shift of +1 LO multiplies the pair odds
by 2.7 regardless of the starting point, which is why LO differences are comparable
across videos and pairs.

**Flip rate**: the fraction of videos of one class for which, after the intervention,
P(opposite class) > P(own class). This is a pair-restricted decision change. Unsteered
base rates are 0.00-0.04 (the occasional baseline misclassification). **Top-1 flip**
is stricter: the full 174-class argmax becomes the opposite class.

### 0.5 The concrete objects: datasets, class pairs, features, directions

**Datasets.** (a) NFP ball datasets: 3,000 Kubric-rendered 16-frame videos of a white
ball on a gray floor; each video has a scripted velocity profile (family A: fixed
heading, varying speed — constant, linear accel/decel, sinusoidal, step, slow-fast-
slow; family B: fixed speed, varying heading — gradual turn, sharp turn,
back-and-forth) and a start position drawn uniformly and independently. v1 samples
profiles uniformly; v2 uses the LP-selected profile mix (section 2.0). (b) Steering
readouts use real SSv2 validation videos, decoded from webm, 16 uniform frames.

**The temporal class pairs** (exact SSv2 labels; short key used in tables):
- push_lr: "Pushing [something] from left to right" vs "Pushing [something] from
  right to left"
- pull_lr: "Pulling [something] from left to right" vs "Pulling [something] from
  right to left"
- cam_lr: "Turning the camera left while filming [something]" vs "Turning the camera
  right while filming [something]"
- move_ud: "Moving [something] up" vs "Moving [something] down"
- cam_ud: "Turning the camera upwards while filming [something]" vs "Turning the
  camera downwards while filming [something]"
- fall_speed: "[Something] falling like a rock" vs "[Something] falling like a
  feather or paper"
- spin_stop: "Spinning [something] so it continues spinning" vs "Spinning [something]
  that quickly stops spinning"
- Depth pairs, used only as negative controls (no NFP variable matches them):
  "Moving [something] towards the camera" vs "... away from the camera"; "Moving
  [something] closer to [something]" vs "Moving [something] away from [something]";
  "Approaching [something] with your camera" vs "Moving away from [something] with
  your camera".

**The 11 feature-dependent ("family") classes**, identified empirically in section
1.4 as the classes where erasing the NFP span does at least 0.4 more damage than a
random span: "Turning the camera right while filming [something]", "Turning the
camera left ...", "Turning the camera upwards ...", "Turning the camera downwards
...", "Lifting up one end of [something] without letting it drop down", "Tilting
[something] with [something] on it until it falls off", "Lifting a surface with
[something] on it but not enough for it to slide down", "Poking [something] so
lightly that it doesn't or almost doesn't move", "Pushing [something] so that it
slightly moves", "Spinning [something] so it continues spinning", "[Something]
colliding with [something] and both are being deflected".

**Named features.** Features are indexed feat00000-feat06143 in the 6,144-feature
SAE. The frequently referenced subsets:
- The 12 pair-screen flippers (>= 50% flips on >= 1 pair, section 1.1): feat00214,
  feat00250, feat00294, feat00950, feat01262, feat01321, feat02818, feat03968,
  feat04254, feat04562, feat04665, feat05077.
- The 9 bidirectional motion knobs from the earlier clamp program (Steering_Experiments_Report.md):
  feat01321 [vel_y], feat02509 [accel_mag], feat03968 [vel_x], feat00950 [vel_x],
  feat04665 [vel_y], feat01699 [vel_x], feat03672 [speed], feat00214 [vel_x],
  feat03059 [vel_x]. Five of them are also pair-screen flippers.
- Example own-concept assignments (the data-defined concept from top-25 activating
  clips, section 1.1a): feat02312 [speed] -> "Spinning [something] that quickly stops
  spinning"; feat03714 [vel_x] -> "Pushing [something] so that it slightly moves";
  feat04280 [vel_y] -> "Lifting [something] with [something] on it"; feat00250
  [vel_x] -> "Spinning [something] so it continues spinning"; feat02211 [speed] ->
  "Poking [something] so it slightly moves".
- v1's two direction-tagged features: feat02825, feat05216. On the v2 dataset the
  direction-dominant features are three new ones: feat00115, feat01217, feat03747.

**Explicit directions and how each is obtained.**
- c_bar_tau (one per kinematic variable): the mean within-video covariance vector
  between the 768-d ball-token activation and the variable, computed in closed form
  from the cached ball activations (formula in section 2.1); unit-normalized before
  steering. v1 and v2 versions differ only in which dataset the covariance is taken
  over.
- head-diff (per pair): (W_head)_cA - (W_head)_cB, the difference of the two classes'
  rows of the model's final linear classifier, unit-normalized. Read directly from
  model weights; the theory identifies it as the per-unit-norm optimal steer for the
  pair, and it serves as the ceiling in every direction experiment.
- Supervised class axis (context in section 4): v = mean layer-11 residual over 24
  clips of class A minus the mean over 24 clips of class B. Decomposing this axis
  onto the SAE decoder columns is how the object-direction-carrying features outside
  the flag set were found (top carriers: feat05672 for pushing right-to-left,
  feat00255 for pulling, feat01652, feat01387 for "Moving [something] up",
  feat00860, feat05258); those features are supervised discoveries and are cited as
  context, not as NFP evidence.
- Null directions (section 2.1 ablations): 20 uniform-random unit vectors in 768-d,
  and 10 manifold-matched vectors built as random unit combinations of the top-50
  principal components of the ball-token activation matrix.

## Part 1. Evidence built on the v1 NFP feature set (85 features)

### 1.1 Population steering and the unselected pair screen

Scripts `steer_concept_alignment.py`, `steer_pair_screen.py`; results
`expA_concept_alignment_s100.json`, `expA_concept_alignment_s40.json`,
`expB2_pair_screen.json`, `expB2_controls.json`.

**Design.** Two levels of the same question: does clamping the flagged features move
classification toward temporal content?

(a) Own-concept steering. Each feature's concept is defined by the data: run 384 real
validation clips, record the feature's mean-pooled activation per clip, take its top-25
activating clips, and use the normalized histogram of their ground-truth classes as a
concept weight vector w over the 174 classes. Clamp the feature at strength s on all
tokens; measure alignment = dP . w, where dP is the mean change in the class
probability vector. Alignment is z-scored against 500 random permutations of the class
labels in w: z = (alignment - null mean) / null sd. Meaning of z: how many standard
deviations the probability mass moved toward the feature's own classes, beyond what an
arbitrary class set would show. The population question: do the 85 features have higher
alignment-z than 30 random features run through the identical procedure?

(b) Pair screen. Every one of the 85 features, with no selection, is clamped at s =
+150 and -150 on 12 videos per side of 10 temporal pairs. Per (feature, pair):
**shift** = [LO(+150) - LO(-150)] / 2 averaged over both input sides. Meaning: the
feature's two-way leverage on that pair's axis; sign says which way +s pushes. Also the
best-orientation flip rate (using +s toward one class and -s toward the other,
whichever orientation the measured shift indicates).

**Parameters.** (a) s in {40, 100}; 384 clips; 500 label permutations; 30 random
features as population null; seed 0. (b) 10 pairs (3 horizontal object/camera, 2
vertical, 3 depth, 1 speed, 1 spin), 12 videos/side, clamps +/-150.

**Intended effect.** If the flags mean something temporal, features should move
probability mass toward their own data-defined concepts more than random features do,
and pair-screen effects should organize by each feature's kinematic tag (a vel_x
feature should move horizontal pairs, not vertical ones).

**Observation.** (a) Own-concept steering, population comparison:

| group | mean alignment-z (s=100) | Mann-Whitney p vs random |
|---|---|---|
| 85 NFP features | +2.29 | 2.5e-4 (s=100), 2.1e-3 (s=40) |
| 30 random features | +0.10 | - |

The Mann-Whitney p is the probability of this rank separation if both groups came from
one distribution. Per-feature z correlates r = 0.97 (Pearson) between s=40 and s=100:
the effect is a stable property of specific features, not an artifact of one clamp.

(b) Pair screen, the strongest per-feature results (best-orientation flip rate; base
rates 0.00-0.04):

| feature | tag | pair | flip rate |
|---|---|---|---|
| feat00250 | vel_x | spin_stop | 1.00 |
| feat04665 | vel_y | cam_ud | 0.79 |
| feat01321 | vel_y | cam_lr | 0.75 |
| feat00214 | vel_x | cam_lr | 0.71 |
| feat05077 | vel_x | fall_speed | 0.67 |
| feat04562 | speed | cam_ud | 0.67 |
| feat02818, feat04254 | speed | fall_speed | 0.62 |

feat00250's own-concept class from (a) was literally "Spinning [something] so it
continues spinning". Tag organization (a pair is axis-matched to a feature if its
temporal property equals the feature's dominant tag):

| comparison | mean \|shift\| | test |
|---|---|---|
| axis-matched (feature, pair) cells | 0.64 | Mann-Whitney p = 3.9e-5 |
| axis-mismatched cells | 0.40 | - |

Only vel-tagged features flip camera-pan pairs at all (speed- and accel-tagged: 0.00).

**Ablations.**

| control | result | comparison |
|---|---|---|
| 30 static features, identical screen | mean \|shift\| 0.335; **0/30 reach 50% flips** | 85 NFP: 0.447 (MW p = 3.9e-3); 12/85 flippers (Fisher p = 0.021) |
| 3 depth pairs (no matching NFP variable) | NFP features flip them at 0.039 | vs 0.094 on all other pairs |

The Fisher test asks how likely a 12-vs-0 flipper split is if flipping were equally
probable in both groups. Interpretation: any hard clamp moves log-odds somewhat, but
crossing decision boundaries is exclusive to the flagged set, and the flagged set is
not a generic disruptor (depth control).

### 1.2 Transplant between videos (interchange intervention)

Script `steer_transplant.py`; results `expC1_transplant.json`, `expE_specificity.json`.

**Design.** Take a receiver video of one direction class and a donor video of the
opposite class, paired one-to-one. Run the donor, record the 85 features' full
token-level activations (1568 x 85 values). Run the receiver, but patch those features
to the donor's values (operation 2; decode + re-add error). Measure **dLO toward
donor** = mean over receiver videos of [LO_patched - LO_baseline], where LO is oriented
as log P(donor class) - log P(receiver class). Meaning: how many log-odds the
transplanted features move the receiver's prediction toward the donor's class; +2.3
LO means the pair odds moved by a factor of 10.

The critical property: every patched value is a real activation from a real video, so
no value leaves its natural range. The saturation (attractor) artifact of hard clamps,
where any feature pushed far enough drags every input to a fixed output, cannot
produce this effect.

**Parameters.** 7 temporal pairs, both transplant directions, 12 videos/side, batch
pairing i-to-i, seed 0.

**Intended effect.** If the features carry the motion percept, moving their natural
activation pattern from video B into video A should move A's prediction toward B's
class, and control features should not.

**Observation and ablations** (dLO toward donor; controls from the specificity
battery):

| pair | NFP-85 | random-85 (s1 / s2) | act-matched-85 | all-6144 ceiling |
|---|---|---|---|---|
| cam_lr | **+3.33** | +0.00 / +0.01 | +0.03 | +9.88 |
| cam_ud | **+2.16** | +0.10 / +0.01 | +0.09 | +8.90 |
| push_lr | +0.05 | ~0 | ~0 | +9.46 |
| move_ud | +0.01 | ~0 | ~0 | +6.56 |
| fall_speed | +0.03 | +0.85* | ~0 | +5.25 |

(*one lucky feature in that random draw; not reproduced by the second seed.)

The all-6144 ceiling also flips the strict top-1 prediction on 81% of videos, so the
method can carry the full percept when given the full code. Reading: on camera pairs
the flagged features carry a quarter to a third of the transplantable percept, 20-30x
every matched control; on object pairs they carry nothing, consistent with the
object-direction axis living in features outside the flag set.

### 1.3 Restoration after destroying temporal structure

Scripts `steer_shuffle_restore.py`, `steer_reverse_play.py`; results
`expC2_shuffle_restore.json`, `expD3_reverse_play.json`, `expE_specificity.json`,
`expE2_direction_nulls.json`.

**Design.** Corrupt the input's temporal structure, then give back only the flagged
features and measure how much of the percept returns.

Corruption 1, shuffling: permute the 8 two-frame blocks of the input video (a fixed
random non-identity permutation per video, shared across conditions). Every frame's
appearance is intact; motion order is destroyed.

Corruption 2, reversal: play the video backward (frame order flipped). Appearance and
speed statistics are exactly identical; only the direction of time changes.

Restoration: run the corrupted video, but patch the chosen features' token activations
back to their values from the clean run.

**Recovery** = (LO_restored - LO_corrupted) / (LO_clean - LO_corrupted), computed on
LO_own = log P(own class) - log P(opposite class). Meaning: 0% = restoring these
features returned none of the destroyed pair discrimination; 100% = all of it. The
all-6144 condition calibrates the method (it should approach 100%), and same-size
control sets calibrate chance.

**Parameters.** Shuffle: 7 pairs, 12 videos/side. Reversal: the 5 direction pairs,
same videos. Seed 0 throughout.

**Intended effect.** If part of the model's temporal percept flows through the 85
features, restoring only them (1.4% of the dictionary) should recover a
disproportionate share of the destroyed discrimination, concentrated on the concepts
the features are tagged with.

**Observation and ablations, shuffling.** Shuffling destroys 1.78 LO on average
(clean +4.15, shuffled +2.38; every pair drops, range 0.7-2.7); this is by
construction the temporal part of the pair signal. Recovery per restored set:

| restored set | share of dictionary | recovery |
|---|---|---|
| all 6,144 (ceiling) | 100% | 100% |
| **NFP-85** | 1.4% | **17%** |
| flippers-12 | 0.2% | 16% |
| random-85 (3 seeds) | 1.4% | 1% / 0% / 4% |
| static-85 | 1.4% | 2% |
| activation-matched-85 | 1.4% | 6% |

Recovery of the NFP set by pair (where it concentrates):

| pair | recovery |
|---|---|
| cam_lr | 65% |
| cam_ud | 39% |
| spin_stop | 35% |
| object pairs (push, pull, move_ud) | ~0% |

The matched control's 6% shows active features carry a little generic restorable
signal; the flagged set triples it and concentrates it on the tagged concepts.

**Observation and ablations, reversal.** The model actively inverts its percept: mean
LO_own +4.79 forward, -3.19 reversed (camera pairs flip at 100%); 7.98 LO destroyed,
a stronger corruption than shuffling and one that isolates a single temporal property.

| restored set | recovery |
|---|---|
| all 6,144 (ceiling) | 86% |
| **NFP-85** | **13%** |
| random-85 (3 seeds total) | 0% / 1% / 0% |
| static-85 | 0% |
| activation-matched-85 | 2% |

A free measurement requiring no intervention: per-feature activation change under
reversal, |mean act forward - mean act reversed| / mean act forward, median per tag
group. Physics dictates what reversal changes: direction inverts; speed and |dv/dt|
(acceleration magnitude) are unchanged.

| tag group | median relative change | physics expectation |
|---|---|---|
| direction | 1.73 | changes most |
| speed | 0.64 | - |
| vel_x | 0.57 | changes (sign flips) |
| vel_y | 0.53 | changes (sign flips) |
| static-85 (reference) | 0.35 | baseline |
| **accel_mag** | **0.05** | **invariant** |

The ordering matches physics exactly, including the near-invariance of the
acceleration features (7x below even the static baseline): the NFP tags predict which
features respond to an arrow-of-time flip, including the required invariance.

### 1.4 Span erasure (necessity)

Script `steer_span_erasure.py`; results `expC4_span_erasure.json`,
`expE_specificity.json`.

**Design.** Build the orthogonal projector onto the span of the 85 decoder columns
and replace every layer-11 token x with x - Px (operation 4). Measure top-1 accuracy
per class on clean inputs. Two methodological points. First, simply zeroing the
features' activations is NOT a valid necessity test: we verified accuracy is exactly
unchanged under zeroing, because zeroing removes only the current positive activations
while the model can still read residual signal along the directions. Erasing the span
removes the channel itself. Second, an 85-dimensional span is 11% of the 768-d space,
so erasing any such span costs something; the comparison against same-size control
spans isolates the features' specific contribution.

**Parameters.** 10 validation videos per class, all 174 classes; projector by QR of
the 85 unit decoder columns; controls are projectors built from random-85 (two seeds
across runs plus the original run's seed), static-85, and activation-matched-85
columns.

**Intended effect.** If the model reads its temporal concepts from these directions,
their removal should selectively destroy the classes matching the features' concepts
and leave everything else roughly intact.

**Observation and ablations.** Overall and family-class top-1 accuracy under each
erased span:

| condition | all 174 classes | 11 family classes |
|---|---|---|
| baseline (nothing erased) | 0.629 | 0.764 |
| **erase NFP-85 span** | 0.583 (-4.6 pts) | **0.118** (-0.65) |
| erase random-85 span (seed 1 / seed 2) | 0.590 (-3.9) | 0.764 / 0.682 |
| erase static-85 span | 0.609 | - |
| erase activation-matched-85 span | - | 0.636 |

The 11 family classes are defined empirically as those where NFP-span damage exceeds
random-span damage by >= 0.4 (list in section 0.5); they match the features' concepts.
Per-class selectivity of the destruction:

| class | baseline | NFP span erased | random span erased |
|---|---|---|---|
| Turning the camera right | 1.00 | **0.00** | 1.00 |
| Turning the camera upwards | 0.80 | **0.00** | 0.90 |
| Turning the camera left | 0.90 | 0.30 | 0.90 |
| Lifting up one end of something | 1.00 | 0.10 | 0.80 |
| Tilting something until it falls | 0.80 | 0.10 | 0.80 |
| Poking so lightly it barely moves | 0.70 | **0.00** | 0.50 |
| Pushing so that it slightly moves | 0.60 | **0.00** | 0.40 |
| Spinning so it continues spinning | 0.50 | **0.00** | 0.40 |

The strongest control costs 0.13 on the family classes where the flagged span costs
0.65. Neither subspace size (all spans are 85-dimensional) nor feature activity (the
matched span) explains the destruction; the specific directions do.

### 1.5 Error repair (utility)

Script `steer_error_repair.py`; results `expD2_error_repair.json`,
`expE_specificity.json`.

**Design.** The positive converse of erasure: on videos of the 11 feature-dependent
classes that the model currently gets WRONG, amplify the 85 features (operation 3,
f -> alpha f on all tokens) and count how many errors become the correct class.
**Repair rate** = repaired errors / total errors mined. Collateral checks: (a) the
fraction of previously-correct family videos that stay correct under the same
amplification; (b) top-1 accuracy on a 200-video sample of other classes.

**Parameters.** Errors mined over up to 24 videos per family class: 59 errors total.
alpha in {1.5, 2, 3}. Controls at alpha = 3.

**Intended effect.** If the features encode what these classes require, boosting them
within their natural activation pattern should push borderline errors over the
decision boundary, and matched controls should not; and the boost should not damage
anything else.

**Observation and ablations.** Repair rates (repaired / 59 mined errors):

| feature set | alpha=1.5 | alpha=2 | alpha=3 |
|---|---|---|---|
| **NFP-85** | 10.2% (6) | 13.6% (8) | **16.9% (10)** |
| random-85 (2 seeds, alpha=3) | - | - | 0.0% (0) / 3.4% (2) |
| static-85 (alpha=3) | - | - | 8.5% (5) |
| activation-matched-85 (alpha=3) | - | - | 3.4% (2) |

Collateral checks at the best condition (NFP-85, alpha=3):

| check | result |
|---|---|
| previously-correct family videos staying correct | 99.0% |
| other-class 200-video sample accuracy | 0.645 -> 0.650 (unchanged) |

The dose-monotone curve supports a graded causal effect rather than noise. The
matched control equaling random shows the repair does not come from amplifying active
features per se. Caveat stated with the result: n = 59 errors, so the best comparison
is 10 repaired vs at most 5 for any control; quote counts alongside percentages.

## Part 2. Evidence built on the v2 (decorrelated) NFP dataset

### 2.0 What v2 is and why it exists

The v1 ball dataset has one systematic flaw: the within-video covariance between the
direction variable and vel_y is +0.68 averaged over the dataset (the variables have
variance ~1.0, so this is a large coupling). It survives heading randomization for a
parity reason: over the circle, the integral of theta * sin(theta) is positive, while
theta * cos(theta) integrates to zero, so direction-vel_y cannot be cancelled by
randomizing headings while direction-vel_x can. Consequence: any quantity derived
from v1 covariances mixes direction with vel_y.

v2 regenerates the 3,000 videos with velocity profiles selected by a linear program
over a 4,000-profile pool so that ALL 10 pairwise covariances among the five
variables are zero at the dataset level (achieved residuals <= 0.0022 in the rendered
set, a 1500x reduction of the direction-vel_y coupling; verified from the rendered
metadata). Start positions are drawn from the same per-video RNG as v1, so v2 differs
from v1 only in the velocity profiles. On v2 the NFP test flags 109 features (70 of
the v1 85 re-flagged), and the recovered concept directions disentangle: the cosine
between c_bar[direction] and c_bar[vel_y] drops from 0.96 to -0.17 (see 2.1 for
c_bar).

### 2.1 Steering the maximum-covariance direction c_bar[vel_x]

Scripts `steer_cbar_direction.py`, `controls_direction_nulls.py`; results
`expF1b_cbar_v2_steering.json`, `expE2_direction_nulls.json`.

**Design.** For a linear readout a(V,t) = w^T h(V,t) along a direction w, the theory
note (section 2) shows the direction maximizing the expected within-video covariance
with concept tau is the mean covariance vector itself:

  c_bar_tau = mean over videos of (1/T) sum_t (h(V,t) - mean_t h)(tau(V,t) - mean_t tau),

a 768-d vector computable in closed form from the cached ball-token activations.
Meaning: c_bar is the optimal DETECTION direction for tau under the stimulus. Whether
it also STEERS (i.e. whether the classifier reads that direction for motion
judgments) is the empirical question this experiment answers. Steering: add delta *
unit(c_bar) to all 1568 tokens (operation 5) and measure the pair shift = [LO(+150) -
LO(-150)] / 2 and the best-orientation flip rate on the concept's matched pairs. The
per-unit-norm ceiling is the head-difference direction (the difference of the two
classes' rows of the classification head), which the theory identifies as the optimal
steer for a pair.

**Parameters.** c_bar computed from the v2 activations (3,000 x 8 x 768, off-screen
steps zeroed as in the NFP pipeline). delta in {-150, -50, +50, +150}; 12 videos per
side; matched pairs per concept (vel_x: camera and pushing left/right; vel_y: camera
and moving up/down; speed: falling pair; accel: spinning pair).

**Intended effect.** If the covariance-derived direction is one the classifier
actually uses, steering along it should move its matched pairs; if detection and
steering directions differ (as they did at the feature level), it should not.

**Observation.** Per direction and matched pair (shift, best-orientation flips,
strict top-1 flips; head-diff and random directions at the same unit norm):

| direction | pair | shift | flips | top-1 |
|---|---|---|---|---|
| **c_bar[vel_x] (v2)** | cam_lr | **-4.08** | **0.42** | **0.25** |
| c_bar[vel_x] (v1, contaminated) | cam_lr | -3.80 | 0.29 | 0.12 |
| c_bar[vel_y] (v2) | cam_ud | +3.24 | 0.25 | 0.00 |
| c_bar[direction] (v2) | cam_lr | +1.48 | 0.00 | 0.00 |
| c_bar[vel_x] | push_lr | +0.15 | 0.00 | 0.00 |
| c_bar[vel_y] | move_ud | -0.11 | 0.08 | 0.04 |
| c_bar[speed] | fall_speed | -0.59 | 0.12 | 0.04 |
| c_bar[accel_mag] | spin_stop | -0.47 | 0.33 | 0.12 |
| head-diff (ceiling, every pair) | all | +11.3 to +13.3 | 1.00 | 1.00 |

The shift sign of c_bar[vel_x] says +delta pushes toward camera-right. Decorrelating
the stimulus sharpened the recovered direction's steering (flips 0.29 -> 0.42, top-1
0.12 -> 0.25). The E[speed] dose-response of c_bar[speed] on neutral videos is flat
(-0.07 to -0.00 over the full delta range). cos(c_bar, head-diff) <= 0.27 on every
matched pair: detection-optimal directions are not the directions the head reads, and
this persists under perfect stimulus statistics — the direction-level form of the
detection-vs-steering split seen at the feature level.

**Ablations.** The 30-direction null battery (empirical p = fraction of null
directions with |shift| >= the c_bar value):

| pair | c_bar \|shift\| / flips | null mean \|shift\| | null max \|shift\| | null max flips | empirical p |
|---|---|---|---|---|---|
| cam_lr | 4.08 / 0.42 | 0.60 | 2.52 | 0.12 | **0/30** |
| cam_ud | 3.24 / 0.25 | 0.59 | 2.90 | 0.21 | **0/30** |

The manifold nulls matter: uniform-random directions average |shift| ~ 0.3 because
they barely intersect the model's active subspace; manifold-matched directions average
1.1 (max 2.52 / 2.90), about 2x stronger, and are the honest "any realistic direction"
null. No null direction reaches the c_bar effect on either pair; the cam_ud margin
over the best manifold null is the thinner of the two.

## 3. Summary table

| evidence | type | key number | how calculated | strongest control |
|---|---|---|---|---|
| own-concept steering | sufficiency, population | MW p = 2.5e-4 | alignment-z of 85 vs 30 random features | 30 random features + 500 label permutations |
| pair screen | sufficiency, tag-organized | 12/85 vs 0/30 flippers, Fisher p = 0.021; axis-match p = 3.9e-5 | per-feature two-way LO shifts and flips over 10 pairs | 30 static features, depth pairs as internal negative |
| transplant | sufficiency, natural values | +3.3 / +2.2 LO vs <= +0.10 | dLO toward donor after patching donor activations | activation-matched-85 |
| restoration (shuffle) | sufficiency | 17% vs 0-6% recovery | (restored - corrupted)/(clean - corrupted) on pair LO | activation-matched-85 |
| restoration (reversal) | sufficiency + tag physics | 13% vs 0-2%; accel invariance 0.05 | same recovery; median relative activation change per tag | 2 random seeds + activation-matched |
| span erasure | necessity | family acc 0.764 -> 0.118 vs >= 0.636 | top-1 accuracy under x - Px | activation-matched span |
| error repair | utility | 16.9% vs <= 8.5% (10/59 vs <= 5/59) | repaired errors / mined errors, alpha-monotone | static-85, activation-matched |
| c_bar[vel_x] steering (v2) | direction-level | 42% flips; p = 0/30 vs nulls | two-way LO shift of a unit direction | 10 manifold-matched directions |

## 4. Scope, stated with the claims

All positive effects concern the concept family a ball probe can see: global and
camera motion, slight motion, spinning, vertical manipulation. Three boundary results
define the limits and belong in any writeup:
1. The features are not necessary for overall SSv2 performance: frame shuffling costs
   the model 30 points of top-1 accuracy across all 174 classes, and restoring the 85
   recovers 3% of that, no better than controls. Their causal role is the concept
   family, not the model's temporal capability at large.
2. The object-level direction axis (pushing left vs right) is fully steerable at
   layer 11 (a supervised direction flips it at 100%), but it is carried by features
   that never activate on ball videos; no ball-derived object — feature set or
   c_bar direction — moves it.
3. Feature identity does not transfer across SAE architectures (median cross-
   dictionary cosine 0.31), but the flagged spans share a significant common core
   (span overlap 0.336 vs 0.209 for random spans, ~10 sd). Durable claims should be
   made about the temporal subspace and the concept family, not individual feature
   indices.
