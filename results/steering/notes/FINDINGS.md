# Experiment A — steering temporal SAE features in VideoMAE: findings

> Local working note (git-ignored). Companion to PROPOSAL.md. Autonomous session log.
> Setup: ReLU SAE (6144 feats) on layer-11 post-MLP residual of
> `MCG-NJU/videomae-base-finetuned-ssv2` (174-way action classifier). Steering recipe:
> clamp one SAE feature to value s on all 1568 tokens (encode -> set f_k=s -> decode -> re-add
> reconstruction error), finish the forward pass, read the SSv2 class distribution. Inputs:
> real SSv2-validation clips (decoded locally from SSv2/videos). Baseline top-1 acc = 68.4%
> on a 256-clip sample (sane). All results in local_runs/steering/*.json.

## The arc (honest, in order)

### 1. Two-feature pass on a hand-built FAST/SLOW class axis
- feat02818 (speed) at 10x clamp: force_shift z=+4.5 vs random null, top-rising "falling like a
  rock", clean monotone s-sweep + sign-flip. feat05087 (accel): null.
- Looked promising but rested on one feature and an arbitrary class axis.

### 2. Broad sweep of all 85 significant features on the FAST/SLOW axis  -> NULL
- At fixed clamp s=100, general steerability S_f does NOT separate temporal features from random
  (a big clamp moves the output for any feature).
- Directional test: corr(t_speed, z_force) = -0.24, sign-agreement 0.48 (chance). A feature that
  DETECTS fast motion is no likelier to STEER toward fast classes than slow.
- Lesson: a feature's detection direction (encoder) and steering direction (decoder->logits) are
  different vectors and need not agree; and the hand-built FAST/SLOW axis is the wrong projection.

### 3. LLaVA-faithful concept-alignment test  -> POSITIVE  (the real result)
Let the DATA define each feature's concept: the SSv2 classes of its own top-25 activating clips.
Steer the feature; measure whether probability mass moves toward THOSE classes (alignment),
z-scored against 500 label-permutations, then compared across features vs 30 random features.
(s=100, 384 clips; `expA_concept_alignment_s100.json`.)

- **Significant features steer toward their own concept more than random:** Mann-Whitney p=2.5e-4.
  z>=3 in 35% of significant vs 17% of random; mean align-z +2.29 vs +0.10.
- **10/85 exceed the strict random-mean+2sd bar (z>6.7).**
- **Motion-coherent top features (the convincing ones):**
  - feat04280 [vel_y]  z=11.3 -> Lifting / Dropping / camera-downward   (vertical motion)
  - feat03150 [speed]  z= 7.4 -> Rolling down a slant / Rolling on flat   (sustained motion)
  - feat02312 [speed]  z= 6.8 -> Spinning that quickly stops / Hitting    (speed/impact)
  - feat02211 [speed*] z= 6.7 -> Poking/Pushing so it slightly moves      (*fires-on-slow -> low motion)
  - feat00978 [vel_x]  z= 6.3 -> Lifting / Moving up
  - feat03714 [vel_x]  z= 8.5 -> Pushing slightly / Moving up / Poking
- Caveat: some high-z features map to generic "Showing [something] ..." classes (high-prior
  attractors, not motion) — separate these from the motion-coherent winners when reporting.

**Verdict so far:** the steering trick works for a real subset of temporal SAE features, with
concept matches that are physically sensible — not a coincidence (p=2.5e-4 at the population
level; ~10 strong individual features). The crude-axis null in step 2 was a wrong-projection
artifact, resolved by the data-defined own-concept readout.

### 4. Static-input steering (frozen frame -> inject motion)  -> QUALITATIVE WIN
Feed a motionless clip (one real frame repeated 16x) and steer one feature.
(s=100, 256 clips; `expA_static_s100.json`.)
- Baseline on a frozen frame predicts exactly the static classes ("Showing [something] on
  top of/next to/behind", "Holding") -> the input genuinely reads as motionless.
- Aggregate MOTION-vs-NOMOTION keyword score does NOT separate sig from random (sig d_motion
  mean +0.086 vs random +0.106) -- same lesson as step 2: a big clamp moves output for any
  feature, and a broad keyword set lumps all motion together.
- BUT per-feature the effect is striking and concept-coherent -- steering one feature on a
  motionless frame makes the classifier hallucinate THAT feature's specific motion:
  - feat02312 [speed] -> "Spinning that quickly stops spinning"  (+0.18 on that single class) <-- best demo
  - feat00250 [vel_x] -> "Spinning so it continues spinning"
  - feat03714 [vel_x] -> "Pushing so it slightly moves"
  - feat02125 [vel_y] -> "falling like a feather or paper"
- Takeaway: crude aggregate metrics never separate from null; the data-defined own-concept
  readout (step 3) does, and the frozen-frame demo confirms it causally per feature.

### 5. Clamp-strength robustness (s=40 vs s=100)  -> NOT a saturation artifact
Re-ran the concept-alignment at a non-saturating clamp s=40 (`expA_concept_alignment_s40.json`).
- s=40: sig mean_z +2.38, z>=3 36% vs rnd 20%, MW p=2.1e-3.  s=100: mean_z +2.29, MW p=2.5e-4.
- **Per-feature z correlation s40 vs s100 = +0.97** -- the SAME features steer toward their
  concept at both strengths. The effect is a stable property of specific features, not a
  large-clamp artifact. The top-10 lists are essentially identical at both s.

## Verdict (robust)
Steering temporal SAE features toward their data-defined concept WORKS for a real subset of
features: significant features beat random (MW p ~ 1e-3 to 1e-4 at two clamp strengths), the
effect is clamp-robust (r=0.97 across s), and ~10 features steer strongly + motion-coherently.
Crude hand-built axes (force, motion-keyword) never separate from null because a large clamp
moves output for any feature -- the data-defined own-concept readout is the correct test, and
the frozen-frame demo confirms it causally (feat02312: motionless -> "spinning quickly stops").

### 6. Diff-of-means ceiling (AxBench-style control)
Built v = mean(layer-11 residual, FAST-class clips) - mean(SLOW-class clips); added alpha*v to
the residual on a neutral test set (`expA_diffmeans.json`).
- Force_shift rises to **+0.37 at alpha=8** (top class "falling like a rock" +0.10) -- the best
  LINEAR fast-slow direction steers the force axis ~7x harder than any single SAE feature (~+0.05).
- Honest nuance (consistent with AxBench): single SAE features are NOT the most POWERFUL steering
  directions; a supervised diff-of-means beats them on a chosen axis. The SAE's value is
  INTERPRETABILITY -- ~85 pre-labeled temporal directions that each steer to their own concept
  with no supervision, vs one bespoke supervised vector per target.

### 7. Dose-response demo (LLaVA Fig.6 analog)  -> CLEAN
mean P(feature's concept class) vs clamp s, on MOTIONLESS frozen-frame inputs
(`expA_dose_static.json`). 3/5 demo features give a clean monotone "turn the knob -> concept
appears" curve, and they are the motion-coherent ones:
  - feat02312 [speed]  "Spinning that quickly stops"     P: 0.007 -> 0.300  (s 0->150, 43x)
  - feat03714 [vel_x]  "Pushing so it slightly moves"    P: 0.010 -> 0.336  (34x)
  - feat04280 [vel_y]  "Lifting"                          P: 0.005 -> 0.194  (39x)
  - feat03150, feat02211: flat (non-motion / unstable concept mode -- honest non-examples)
On a frozen frame with zero motion, cranking one temporal feature makes the classifier
increasingly predict that feature's specific motion -- the single-neuron steering demo, for
motion concepts. The 3 heroes hold on REAL inputs too (`expA_dose_ssv2.json`):
  - feat02312 0.007->0.251,  feat03714 0.011->0.237,  feat04280 0.017->0.172  (s 0->150)
  - feat00978/feat02818/feat01230: weak/flat on real -- honest non-heroes.

### 8. Scorecard
`steering_scorecard.csv` ranks all 85 features by mean concept-align z (s40,s100) with a
motion-coherence flag: 32/85 reach z>=3; **16 are motion-coherent AND z>=3**. Hero set for
figures: feat02312 (speed->spinning-stops), feat03714 (vel_x->pushing), feat04280 (vel_y->lifting).

### 9. Strict single-top-class steering count (steer toward feature's ONE majority class)
`expA_topclass.json`. For each feature, concept = the single majority GT class of its top-25
clips; steer s=100; measure dP(that one class) vs a random-feature null (random feats steered
toward THEIR own top class: mean +0.012, sd 0.042 -- wide, because a big clamp moves any class).
- dP>0: 41/85.  dP > null_mean+2sd (+0.095): **5/85**.  >=2x base AND >2sd: 5/85.  z>=3: 1/85.
- Winners skew to attractor classes: feat05785 (->"Showing to the camera", 82x), feat03672
  (->"Turning camera left"), i.e. high-prior / camera-motion labels, not clean object motion.
- Lesson: the single top class is a noisy, attractor-biased target; this harsh test passes only
  the very strongest and conflates motion-steering with label-attraction. Motivates step 10.

### 10. Bidirectional 4-cell speed-axis test (content-controlled; the rigorous adjudicator)
Objective per-class speed score = mean Farneback optical-flow magnitude over each of the 174
SSv2 classes (`class_speed.json`; slow end = Holding/Showing-on-top, fast end = Burying/Stuffing/
Throwing + camera-pan classes). Readout = E[speed|pred] = sum_c P(c)*speed_z(c). For each
feature pick C_hi/C_lo = highest/lowest-speed class among its concept classes (require z-gap>=0.5),
hold those classes' real videos as fixed input, sweep clamp s in {-50,0,50,100,150}, fit slope
dE[speed]/ds in each input set. A genuine speed knob raises E[speed] when steered UP from BOTH a
fast and a slow start (both slopes positive); an attractor only pulls toward its fixed point.
(`expA_speed_axis.json`; random null mean-slope -0.0001 +/- 0.0018.)
- **Dominant pattern = regression-to-attractor:** most features have slope_hi<0, slope_lo>0 --
  both inputs CONVERGE to the feature's fixed output point as s grows (sign set by each input's
  distance from it), NOT a speed knob. The crude single-class test (step 9) cannot see this.
- Counts: pass ALL 4 directional cells **5/85**; >=3 cells 24/85; both-slopes-positive 15/85;
  **both-positive AND slope_z>=2 vs random = 9/85 robust speed steerers.**
- The robust 9 (steer up -> more predicted motion from both fast & slow starts): feat01321
  [vel_y] z+7.2, feat02509 [accel] +5.3, feat03968 [vel_x] +5.3 (4/4), feat00950 [vel_x] +5.2,
  feat04665 [vel_y] +3.6 (4/4), feat01699 [vel_x] +3.4, feat03672 [speed] +3.4 (4/4),
  feat00214 [vel_x] +3.1, feat03059 [vel_x] +3.1 (4/4).
- **Honest confound:** flow-based axis makes most features' C_hi a "Turning the camera ..." class
  (camera pan = high global flow), so "more motion" at the top partly = camera/global motion.
  Cleanest object-motion cases: feat03672 (C_hi="Burying", 4/4) and feat04665 [vel_y] (4/4).
  TODO robustness check: rebuild speed axis excluding the 4 camera-pan classes, re-count.
- Takeaway: under a content-controlled bidirectional test, single-feature steering is MOSTLY an
  attractor effect; only a robust minority (~9/85) act as genuine, direction-consistent motion
  knobs. This refines (does not overturn) the positive verdict -- the effect is real but smaller
  and more confounded than the soft concept-alignment count (32/85) suggested.

### 11. Generalization of the proven 9 across UNSEEN input classes (the strongest test)
`expA_generalize.json`. Test C proved each of the 9 on its OWN two anchor classes (C_hi/C_lo from
its concept set). This asks: does the knob persist on classes it was NOT tuned on? Build a
12-class panel spanning the full motion spectrum (speed_z -2.38 "Showing on top" .. +4.28
"Burying"; "Turning the camera" classes excluded). For each proven feature x each panel class,
hold that class's real videos fixed, sweep s in {0,50,100,150}, measure E[speed|pred]. Metric:
up_frac = fraction of panel classes where E[speed]@s150 > E[speed]@s0 (attractor would push DOWN
from fast starts -> up_frac ~0.5; a true content-general knob -> up_frac ~1).
- **All 9 generalize: up_frac 0.92-1.00** (3 at a perfect 1.00: feat01321, feat02509, feat00950)
  vs **random-feature null up_frac 0.61 +/- 0.19** -- the 9 sit ~1.5-2 sd above random.
- Best feature feat01321 [vel_y]: raises predicted motion from ALL 12 classes, incl. BOTH pulling
  classes (Pulling onto +0.016, Pulling out of +0.016), Pouring, Spinning-stops, Showing-on-top.
  Effect only shrinks at the extreme-fast anchor "Burying" (z+4.28, slope +0.004 -- no headroom).
- **Conclusion:** the 9 are not lucky on their own anchors -- they are CONTENT-GENERAL motion
  knobs (steer up -> more predicted motion regardless of the action shown). This rules out
  regression-to-attractor for the 9 (an attractor pushes down from fast starts; these push up from
  all 12) and is a stronger claim than Test C alone. This is the headline positive result.

### 12. Experiment B — direction-pair flipping (partner's proposal; maximally specific demo)
SSv2 has class pairs identical in content, differing only in motion direction: Pushing/Pulling
left-to-right vs right-to-left [vel_x], Moving up vs down [vel_y]. Steer one NFP-tagged
directional feature; try to FLIP the prediction across the pair. Two-way knob via the clamp sign
(s<0 is well-defined since decode is linear in f). Readouts: pair log-odds LO = logP(pos) -
logP(neg) dose-response on both input sides; pair-restricted flip rate; strict top-1 flip; NFP
t-sign vs induced-shift sign agreement; random-feature null. Baselines: model separates all
three pairs at pair-acc 0.96-1.00 unsteered. (`expB_direction_flip.json`.)
- **Up/down: qualitative success.** feat04280 and feat04541 each flip "Moving something down"
  videos to "up > down" in **42% of videos (4% base)**, 10x, monotone LO curves. Top-1 escapes
  to adjacent vertical classes (Lifting etc.) -- the pair-restricted readout is the direction claim.
- **Left/right: single NFP features fail to flip.** Best shifts beat the random null (feat05179
  -2.83 on pulling, feat05987 -1.66 on pushing vs null max ~1.7/0.9) but flips <= 8%.
- **NFP sign does NOT transfer: 11/24 = chance.** Detection direction (encoder, ball-world) does
  not predict steering direction (decoder->logits) -- same lesson as step 2, now on real classes.

### 13. Experiment B ceiling — diff-of-means flips EVERY pair at 100%
v = mean(layer-11 residual | pos-class) - mean(| neg-class), 24+24 build clips; add alpha*v on
all tokens of held-out clips (`expB_direction_diffmeans.json`).
- **All three pairs flip at 100% -- pair-restricted AND strict top-1 -- at |alpha| as small as 2**,
  in both directions, with huge LO swings (~+/-14). Perceived motion direction INCLUDING
  left/right is linearly encoded at layer 11 and fully steerable by a uniform token edit.
- So the single-feature failure on horizontal pairs is a property of the SAE DICTIONARY, not of
  the layer or the uniform-token intervention.

### 14. Mechanism — the direction axes are concentrated in features NFP never flagged
Decompose each pair's v onto unit decoder columns (`expB_axis_decomposition.json`).
- Axes are CONCENTRATED, not distributed: top-1 |cos| = 0.54 (push), **0.77 (pull, feat00255)**,
  0.62 (up/down, feat01387); top-5 columns give R^2 = 0.73 / 0.91 / 0.55; top-20 ~0.82-0.93.
- **None of the top-10 axis-carrying features (any pair) are among the 85 NFP-flagged features.**
  The NFP span captures only 18-38% of each axis (R^2 0.18 push / 0.38 pull / 0.23 up-down).
- Interpretation: the SAE DID learn direction features, but the ball-video NFP test missed them --
  plausibly because real-scene push/pull motion features do not activate on synthetic ball clips
  (out-of-distribution), so they had no variance to covary. NFP's flag set is sufficient for
  ball-like motion kinematics, not exhaustive for all motion encoding in the model.
- Follow-up in flight: steer the axis-carrying features directly (feat00255 etc.) -- if one
  flips its pair, the single-feature direction-flip demo exists, found via axis decomposition.

### 15. Experiment B RESOLUTION — axis-decomposition features FLIP the pairs (the demo exists)
Steered the top-|cos| features from step 14 on their own pairs
(`expB_axisfeats_{push,pull,updown}.json`). Single-feature, two-way clamp, same protocol:
- **Pushing L->R vs R->L: feat05672 flips at 100% pair AND 100% strict top-1** (pos->neg at
  s=-150; neg->pos 100% pair / 54% top-1). feat04398: 92% top-1 neg->pos. feat04956: 83% top-1.
  LO shifts ~10-12 (vs best NFP feature 1.7; diff-means ceiling ~14).
- **Pulling: feat01652 88% top-1 neg->pos, 100% pair pos->neg; feat02731 100% pair.**
- **Up/down: feat01387 & feat05258 ~92% pair / ~80% top-1 neg->pos, ~96% pair pos->neg.**
- **Sign calibration solved: baseline activation preference (mean act on pos-side minus
  neg-side videos) predicted steering direction 12/12** -- these features have massive side
  preferences (|act_pref| up to 37), i.e. they ARE the model's direction detectors. The NFP
  t-sign remains useless for direction (2/4, 3/4, 0/4).
- **The OOD hypothesis is confirmed directly: most of these flip features have t = NaN in the
  ball-video NFP results** -- literally zero activation variance on synthetic ball clips. The
  model's real-scene direction features are silent on the ball dataset, which is exactly why
  NFP could not flag them.
- Net: the maximally specific causal demo EXISTS -- one SAE feature, clamped, flips the model's
  perceived motion direction between content-identical classes at up to 100% strict top-1 --
  with the honest caveat that the feature was located by supervised axis decomposition + the
  SAE dictionary (unsupervised directions, supervised selection), not by the NFP flag set.

### 16. Experiment B2 — unbiased pair screen: ALL 85 NFP features x 10 temporal pairs
Per user's restructure: no feature selection, no classifier-derived directions. Every NFP
feature steered +/-150 on both sides of 10 SSv2 temporal pairs (horizontal push/pull/cam-pan,
vertical move/cam, 3 depth pairs, falling rock-vs-feather [speed], spinning continues-vs-stops
[accel]); 12 videos/side; readouts = signed pair-LO delta between clamps + best-orientation
flip rates. (`expB2_pair_screen.json/.csv`.) Baselines all >= 0.75 pair-acc.
- **13/85 features reach >=50% flips on at least one pair.** Heroes: feat00250 [vel_x] flips
  spinning-continues<->quickly-stops at **100%** (its own step-4 concept was literally
  "Spinning so it continues spinning"); feat04665 [vel_y] cam-up/down 0.79; feat01321 cam-l/r
  0.75; feat00214 [vel_x] cam-l/r 0.71; feat05077 fall-speed 0.67; feat04562 [speed] cam-ud
  0.67; feat02818/feat04254 [speed] fall-speed 0.62.
- **Tau-consistent structure (the pattern sought): axis-MATCHED pairs get mean |LO delta|
  0.636 vs 0.399 mismatched, MW p = 3.9e-5.** Only vel_x/vel_y features flip cam-l/r at all
  (speed/accel: 0.000); vel_y features are best on cam-ud (0.155) and move-ud; speed features
  strongest on fall-speed. Flip-rate means don't separate (p=0.61) -- flips are driven by a few
  strong features; the LO-power metric carries the population signal.
- **Depth pairs = clean internal negative control**: NFP has no depth tau, and NFP features
  flip depth pairs at 0.039 vs 0.094 elsewhere (~2.4x less) -- the features are NOT generic
  pair-disruptors.
- **Cross-experiment consistency: 5 of the proven-9** (feat01321, feat00950, feat03968,
  feat00214, feat04665) are also top pair-flippers here.
- Honest negatives: object-level push/pull direction pairs stay unflippable by NFP features
  (max 0.21) -- consistent with step 14 (those axes live in ball-OOD features); spin_stop is
  somewhat flippable by everything (weakest baseline 0.75, easiest pair).

### 17. Experiment B2 controls — random static steering + temporal ablation
(`expB2_controls.json`.) Static pool = features with finite NFP t (active on ball videos) and
max|t| < 2 across all taus (measurably non-temporal); 30 sampled, identical screen protocol.
**Part 1 (sufficiency — temporal vs static):**
- mean |LO delta| across pairs: temporal 0.447 vs static 0.335, **MW p = 3.9e-3** — temporal
  features push pair axes harder. (Static features do move LO somewhat: a +/-150 clamp on any
  feature perturbs — the old "big clamp moves output" lesson — but they never flip.)
- **Flippers: 12/85 temporal reach >=50% flips on >=1 pair; 0/30 static. Fisher p = 0.021.**
- max-flip means don't separate (0.32 vs 0.27, p=0.15) — flips are all-or-nothing per feature;
  the flipper-count and LO-power stats carry the signal.
**Part 2 (necessity — ablation, clamp sets to 0):** NULL, with one telling exception.
- Zeroing ALL 85 temporal features leaves pair-acc (0.967), top-1 (0.83) EXACTLY unchanged on
  every pair; same for the 12 heroes; random sets identical. The model's temporal-pair
  discrimination does not depend on these features — redundant encoding, consistent with
  step 14 (axes spread across many features; 85/6144 removed leaves the rest).
- Exception: the discrimination MARGIN |LO| drops specifically on the camera pairs when the
  85 are removed (cam_lr 5.26->4.28, cam_ud 5.36->4.89; random-85 sets stay ~4.5) — the same
  pairs our features flip. Consistent direction, small effect.
- Net framing: NFP temporal features are causally SUFFICIENT to shift/flip temporal-pair
  classification (and static features are not), but not NECESSARY for it — the model encodes
  temporal properties redundantly.
- **Why the ablation null is real, not a bug (diagnostic on pair videos):** the 85 features ARE
  active on these clips (2.5% of token-feature slots, mean act 18, max 139) and ablating them
  removes a delta of norm 36 = **17% of the layer-11 token norm** — the intervention is large
  and verified (it erodes cam-pair margins by ~0.5-1.0 LO). Accuracy still doesn't move because
  baseline pair margins are enormous: |LO| ~ 4.5-5.4 (odds 90-220:1); a <=1.0-LO erosion leaves
  every video far from the boundary. Steering at s=150 injects a delta of norm 150 = **71% of
  token norm, coherently along ONE direction on all 1568 tokens** (8x the features' natural
  operating range of ~18) — 4x larger than removing all 85 natural contributions combined, and
  aligned instead of spread over 85 directions. Sufficiency-at-150 + necessity-null is therefore
  quantitatively coherent, on top of the redundancy argument (step 14: axis mass also lives in
  non-NFP features, which ablation leaves intact).

### 18. Necessity test on HARD (near-boundary) videos — a real but tiny necessity gradient
(`expB2_ablation_hard.json`.) Scanned 48 videos/side/pair, kept the 12 with smallest |LO|
margin; hard-subset baselines: mean LO 2.47 overall, down to 0.81-1.23 on move_ud / spin_stop /
fall_speed (vs 4.5-5.4 on random videos). Ablation battery re-run on exactly these + a new
condition: axis-30 = union of step-14 axis-decomposition top-10s (predicted load-bearing).
- Decision flips remain ~zero for every set (0-1 of 240; random sets also occasionally 1).
- Margin damage orders EXACTLY as predicted: axis-30 dLO = **-0.144** > heroes-12 -0.108 ~
  temporal-85 -0.104 >> random sets ~0.00. Per-feature necessity is highest for axis features.
- **Axis-30 damage is statistically real: paired Wilcoxon vs matched random-30, p = 0.016 and
  0.007 (both seeds).** Temporal-85 vs random-85 is a trend only (p = 0.23, 0.051).
- Absolute effects stay small: -0.14 LO on a 2.47 mean margin (~6%). Even near the boundary,
  no 30-85-feature subset of this 6144-feature dictionary is meaningfully load-bearing.
- Completes the sufficiency/necessity story: NFP temporal features are powerful steerable
  levers (sufficiency, exclusive vs static: Fisher p=0.021) while pair discrimination itself is
  deeply redundant -- the axis features show detectable-but-small necessity in the predicted
  order, and nothing shows strong necessity. "Levers, not wires" is the honest framing.

### 19. Attractor theory formalized and validated from weights alone (partner's framing check)
Partner's formalization: steering shifts logit_c by (s - f_k) * alpha_c with alpha_c =
W_head[c] . mu_k, so at large s every input converges to a fixed distribution — an attractor
determined by the PAIR (mu_k, W_head), not by the feature alone. Refinement from our side: the
head applies LayerNorm before the linear readout and LN is scale-invariant, so the attractor is
EXACT at large s: p_attr(k, sign) = softmax(W @ LN(sign * mu_k) + b), computable from weights
with no forward passes; the +s and -s attractors are different distributions (not mirrors).
Computed for all relevant features (CPU only):
- **Theory predicts measurement: sign(attr_Espeed(+) - attr_Espeed(-)) matches the measured
  4-cell slope sign for 63/77 features (82%).** From weights alone.
- **Group signatures:** proven-9 knobs: attractor sharpness 0.64, |dEspeed(+/-)| = 2.36 (large
  graded speed contrast — the "graded alpha" knob story). Other NFP: 0.31 / 0.46. expB flip
  features: sharpness **0.78**, dEspeed only 0.27 — they are SHARP attractors aimed at
  direction classes (feat05672 +s attractor = "Pushing right to left" at p=0.94; feat00255 =
  "Pulling right to left" p=0.74; feat01387 = "Moving up" p=0.76).
- Consequences for framing, both partner objections CONFIRMED: (a) attractor-vs-knob is a
  property of (mu_k, W_head, readout axis), not intrinsic to the feature; (b) attractors are
  not deceptive per se — our own 100% top-1 direction flips (step 15) ARE attractor behavior
  pointed at the desired class, i.e. useful attractors; and feat02312's +s attractor is its own
  concept class ("Spinning that quickly stops", p=0.42), so the hero dose-response demos are
  partly attractor-driven toward the correct concept. The step-10 language "regression-to-
  attractor masquerades as steering" should be revised to: the attractor pattern is not
  evidence of a motion-specific causal role wrt the chosen axis; it reflects alignment between
  the decoder column and the classification head.
- Connection to literature (from partner's search): Arad et al. (EMNLP 2025) input-vs-output
  feature distinction; SAE-TS (Chalnev et al. 2024) encoder-detects-X / decoder-steers-Y
  mismatch. Our NFP-flag-vs-steering-power split (detection != steering; flip features are
  ball-OOD) is the classification-setting instance, with the 4-cell test as the operational
  separator.

### 20. Input-score vs output-score scatter (Arad-style test)
`figures/input_output_scatter.png`. Input score = NFP detection strength (max |t| across taus).
Output scores = three independent steering measurements: own-concept alignment z, 4-cell slope
z, pair-screen mean |LO delta|. All 85 NFP features.
- **Detection strength and steering power are statistically INDEPENDENT**: pearson r = +0.07 /
  -0.00 / +0.08, all p > 0.26; spearman the same; top-quartile co-occurrence 6/8/7 features vs
  ~5.3 expected under independence.
- Nuance vs Arad et al.: they report high input and output scores "rarely co-occur"
  (anti-correlated); in our classification setting co-occurrence is exactly at CHANCE — the
  input score carries zero information about steering power, in either direction.
- Anecdote: the strongest detector (feat05179, max|t| = 13.2) has ~zero output on all three
  metrics; none of the proven-9 knobs is among the top detectors (their max|t| span 5-9.5).
- For the paper: this is the quantitative form of "detection direction != steering direction",
  and motivates selecting steering features by output-side tests (4-cell, pair screen, or the
  step-19 attractor spectra computed from weights) rather than by detection statistics.

### 21. Experiment C1 — feature transplant (interchange intervention; natural activations)
Motivated by activation-patching literature (Zhang & Nanda 2309.16042; Heimersheim & Nanda
2404.15255; Marks et al. 2403.19647): swap the 85 NFP features' NATURAL token-level
activations from a donor video of the opposite direction class into a receiver (new
SteerLayer patch mode: f[..., idx] <- donor values, decode + re-add error). No clamping, so
no attractor objection. 7 temporal pairs, both directions, 12 videos/side.
(`expC1_transplant.json`.)
- Ceiling validates the paradigm: transplanting ALL 6144 features moves the prediction to the
  donor class at 88% pair / 81% top-1 (mean dLO +7.3).
- **Camera pairs: the percept travels with the NFP features.** cam_lr dLO->donor = +3.33,
  cam_ud +2.16, vs random-85 and static-85 at ~0.01-0.05 (~200x controls; a quarter to a third
  of the full-code ceiling). Flippers-12 carry nearly all of it (+3.22 / +1.57).
- Object pairs (push, pull, move_ud, fall_speed): NFP transplant carries ~nothing — consistent
  with step 14 (object-direction lives in ball-OOD features).
- Flip rates stay low (~8%): 3 LO against a 9 LO ceiling does not cross the boundary. The
  claim is "carries a substantial minority share", which is what redundancy predicts.

### 22. Experiment C2 — shuffle-and-restore (causal tracing)
Corruption: permute the 8 tubelet blocks of the input (appearance intact, motion destroyed —
standard temporal corruption, cf. 2508.11576). Restoration: run the shuffled video but patch
the chosen features' CLEAN token activations at layer 11. Recovery = (restored - shuffled) /
(clean - shuffled) on pair log-odds. (`expC2_shuffle_restore.json`.)
- Shuffling destroys 1.78 LO of pair discrimination on average (clean 4.15 -> 2.38); every
  pair drops (0.7-2.7). VideoMAE genuinely uses temporal order.
- Restoring ALL 6144 -> 100% recovery (mechanism validated).
- **Restoring the 85 NFP features (1.4% of the dictionary) recovers 17% of the destroyed
  signal — 12x their proportional share. The 12 flippers alone (0.2%) recover 16% — 80x.**
  Random-85: 4%. Static-85: 2%.
- Per-pair, recovery concentrates where the features live: cam_lr 65%, cam_ud 39%, spin_stop
  35%; object pairs ~0%. Same dissociation as C1 from an independent paradigm.
- Headline sentence for the paper: restoring 12 NFP-identified features recovers a third to
  two-thirds of the temporal percept that frame shuffling destroys on camera-motion and
  spinning pairs, at 0.2% of the dictionary; matched random and static sets recover ~nothing.

### 23. Experiment C3 — global temporal-capability restoration: honest negative
Same shuffle-and-restore paradigm as C2, measured on top-1 accuracy over 1740 videos, 10 per
class, ALL 174 classes. (`expC3_global_restore.json`.)
- Shuffling costs 0.299 top-1 (0.629 -> 0.330): a large, real measure of the model's overall
  temporal capability. Restoring ALL 6144 features recovers 94% (mechanism validated).
- **Restoring NFP-85 recovers 3% — equal to random-85 (3%) and static-85 (2%). No temporal-
  tercile specificity (1% on the most shuffle-sensitive classes).**
- Reconciliation with C2 (17% overall, 65% cam_lr): the NFP features' temporal content is real
  but NARROW — global/camera motion and spinning, i.e. exactly ball-like motion. SSv2's
  overall temporal capability rests on a much broader inventory (object trajectories,
  hand-object interaction order) that lives elsewhere in the dictionary. 30 points of
  shuffled-away accuracy dwarf the slice these 85 features carry.
- Claim discipline for the paper: NFP certifies concept-level causal relevance (four
  independent demonstrations on matched pairs), NOT global importance to model performance.
  State both.

### 24. Experiment C4 — span erasure: necessity, finally, and class-selective
Erase the subspace spanned by the 85 decoder columns (project x - Px at every token; QR
projector) vs a dimension-matched random-85 span and static-85 span, on the same 1740-video
all-class sample, clean inputs. (`expC4_span_erasure.json`.)
- Overall: NFP-span -4.6 pts vs random-span -3.9 pts — modest globally, consistent with C3.
- **Class-selective destruction exactly on the features' concepts**: Turning-camera-right
  1.00 -> 0.00 (random-span: 1.00), camera-up 0.80 -> 0.00, camera-left 0.90 -> 0.30,
  lifting-one-end 1.00 -> 0.10, tilting-until-falls 0.80 -> 0.10, poking-barely-moves
  0.70 -> 0.00, pushing-slightly 0.60 -> 0.00, spinning-continues 0.50 -> 0.00. These are the
  same concepts identified by steering (B2), transplant (C1), and restoration (C2).
- Why zero-ablation (steps 17-18) missed it: zeroing removes only the current positive
  activations; the directions still carry readable signal. Erasing the span removes the
  channel itself.
- **Revised final framing ("levers, not wires" is now too weak):** for their own narrow
  concept family — global/camera motion, slight motion, spinning, vertical manipulation —
  the 85 NFP features are BOTH sufficient (steering/transplant/restoration) AND necessary
  (span erasure destroys those classes selectively). At the whole-task level they carry
  little (C3: 3% global recovery; C4: -0.7 pts net vs matched span). The features are the
  model's dedicated machinery for a specific slice of temporal perception — the slice the
  ball probe tested — and that claim is now supported from every causal direction.

### 25. Experiment D3 — reverse-play: the model flips its direction percept; NFP features
carry a measurable share; and the tags predict physics-correct invariances
Design: play direction-pair videos backward (frame order flipped; appearance and speed
statistics identical, only the arrow of time changes). Restore forward-run feature
activations into the reversed run (token patch). 5 direction pairs, 12 videos/side, seed 0.
(`expD3_reverse_play.json`, script `steer_reverse_play.py`.)
- **Reversal flips the model's percept almost completely**: mean LO_own +4.79 forward ->
  -3.19 reversed (rev-flip rates 0.75-1.00; both camera pairs 1.00). The model reads a
  reversed video as the opposite direction class. Corruption magnitude 7.98 LO — larger and
  cleaner than shuffling.
- Restoration: NFP-85 recovers **13%** of the destroyed signal; random-85 and static-85
  recover exactly **0%**; ALL-6144 86% (ceiling). Same narrow-but-real pattern as C2.
- **Physics-signature result (new, strong):** per-feature activation change under reversal,
  grouped by NFP tag: direction-tagged features change most (median rel change 1.73),
  speed/vel 0.53-0.64, static 0.35, and **accel_mag features 0.05** — near-invariant.
  This is exactly right physically: |dv/dt| is invariant under time reversal, so genuine
  acceleration features SHOULD not respond to it, and ours do not (7x below even the static
  baseline). The NFP tags predict which features respond to an arrow-of-time flip, including
  the invariance. This is evidence the tags carry physical meaning, obtained without any
  steering at all.

### 26. Experiment D1 — playback-speed mediation: null, for an instructive reason
Design: 1x = 16 frames over the middle half of the clip, 2x = 16 frames over the full span
(same center, doubled stride); 300 videos >= 40 frames; readout E[speed] = P . speed_z;
mediation by patching features from the 2x run into the 1x run.
(`expD1_speed_mediation.json`, script `steer_speed_mediation.py`.)
- **The manipulation barely moves the model: total effect +0.023 +/- 0.014 E[speed] units.**
  With TE ~ 0 the mediated fractions are noise (-5% to +5%; ceiling 69% of a tiny number).
- Diagnosis: SSv2 action categories are mostly speed-invariant by design (pushing at 2x is
  still pushing), so the class-level E[speed] readout is insensitive to playback speed for
  generic videos. The activation-level signature is still directionally right (speed-tagged
  features +0.002, accel-tagged +0.017, static -0.001 from 1x to 2x).
- Salvage designed (D1b): use the one class pair whose IDENTITY is a speed judgment —
  "falling like a feather" vs "falling like a rock". Speeding up a feather video should move
  the prediction rock-ward; then ask whether that shift is mediated by the NFP features.
  Queued behind D2/D4.

### 27. Experiment D2 — error repair by feature amplification: positive, monotone, no collateral
Design: 11 concept-family classes (C4 excess >= 0.4). Mined 59 model errors over up to
24 videos/class. Amplify feature sets within their natural pattern (capture f, patch back
alpha*f; alpha in {1.5, 2, 3}); measure repaired errors (top-1 becomes true class);
collateral on 200 previously-correct family videos and 200 other-class videos.
(`expD2_error_repair.json`, script `steer_error_repair.py`.)
- **Repair rate, NFP-85: 10.2% -> 13.6% -> 16.9% (monotone in alpha).** Controls at alpha=3:
  random-85 5.1%, static-85 8.5%, flippers-12 5.1%.
- **Collateral is clean**: previously-correct family videos stay correct at 99.0%;
  other-class accuracy 64.5% -> 65.0% (unchanged).
- Claim: moderately boosting the identified features corrects ~1 in 6 of the model's errors
  on its feature-dependent temporal classes, at 2-3x the best matched control and zero
  measured cost elsewhere. Small n (59 errors; 10 vs 5 repairs at best), so quote with
  counts; a second seed for the controls would firm it up.

### 28. Experiment D4 — SAE-bottleneck probe: sufficiency as a representation, family-selective
Design: cached mean-pooled layer-11 representations for 4,300 videos (25/class, all 174
classes): raw 768-d residual and 6,144-d SAE features. Logistic probes on frozen features;
train/test split within class (18/7 at the largest size); data-efficiency at 3/8/18
training examples per class. (`expD4_bottleneck_probe.json`, cache `expD4_probe_cache.pt`,
script `probe_bottleneck.py`.)
- **Family task (the 11 feature-dependent classes): a probe on ONLY the 85 NFP features
  reaches 0.72 / 0.83 / 0.78 accuracy at 3/8/18-shot — vs random-85 0.38-0.49, static-85
  0.57-0.73, and the full-residual ceiling 0.91-0.94.** At 3-shot, the 85 interpretable
  features beat matched static features by 15 points and random by 34.
- All-174 task: NFP-85 is NOT special (0.22 vs static 0.25 at 18-shot) — the same
  narrowness as C3, now on the readout side.
- Selectivity: the 174-way NFP-85 probe scores 0.592 on family classes vs 0.195 elsewhere
  (3x) — the feature set is informative specifically about its concept family.
- Literature caveat stated in design (2502.16681): raw-activation probes win overall, as
  expected; the claim is subset sufficiency for the family, not probe superiority.

### 29. Experiment D1b — speed-pair mediation: the instrument fails, informatively
Design: rock-vs-feather LO readout on feather and rock videos at 1x vs 2x playback;
mediation via feature patching from the 2x run. 30 videos/side.
(`expD1b_speed_pair_mediation.json`, script `steer_speed_mediation_pair.py`.)
- **The manipulation itself is dead: TE = -0.09 +/- 0.21 LO on feather videos (wrong sign,
  noise), +0.13 +/- 0.10 on rock.** Speeding a feather video 2x does not make the model see
  a rock-like fall. Likely reason: the rock/feather distinction rests on trajectory
  character (straight accelerating drop vs oscillating drift), which resampling preserves.
- Combined with D1: VideoMAE-SSv2 predictions are largely playback-speed-invariant at 2x,
  both globally and on speed-defined classes. Input-level playback manipulation is a weak
  instrument for this model; mediation through it is unanswerable at 2x. The causal case
  for the speed features rests on the intervention paradigms (dose-response, restoration,
  span erasure), which do not need this instrument.
- If revisited: needs 4x+ factors (frame counts rarely permit) or frame interpolation for
  slow-down; or optical-flow-magnitude readouts instead of class LO.

### 30. Experiment D5 — full-dictionary attribution scan: convergent at group level,
unreliable per-feature
Design: attribution patching (AtP, 2403.00745) of the pair-interchange objective for all
6,144 features at once — donor-capture pass + one gradient on f at layer 11 (AttrLayer
re-expresses the layer output as decode(f)+e with f a grad leaf). 7 pairs, both
directions, 12 videos/side; per-pair scores normalized by pair max, averaged.
(`expD5_attr_scan.json`, script `attr_temporal_scan.py`.)
- **Group-level convergence is strong:** 28/30 of the step-14 axis features land in the
  top-200 attributed (of 6,144), and the per-pair top-5 lists reproduce the axis
  decomposition's discoveries (feat05672, feat01652, feat00255, feat01387, ...) from an
  entirely independent method (gradients vs supervised diff-of-means geometry).
- NFP features score far above the dictionary at large (AUROC 0.789, p = 2.4e-20) but are
  NOT the top drivers of pair discrimination: 8/85 in the top-85, median rank 1067.
  Consistent with the established picture (pair-direction lives in real-scene features).
- **Caveat that limits per-feature claims: the exact-vs-attribution validation on cam_lr
  came out NEGATIVELY correlated (spearman -0.43, p = 0.017).** The linear AtP
  approximation is untrustworthy per-feature in this setting (LayerNorm head +
  saturation are the likely causes). Use the scan as a group-level screener only; a real
  "NFP recall" number would need exact patching of the top-N, which is future work.

### 31. Experiment D7 — the direction code composes (superposition)
Design: steer feat01321 (+s attractor "Turning the camera right", p=0.97) and feat04665
(+s attractor "Turning the camera downwards", p=0.81) separately and jointly on 24 neutral
videos (camera classes excluded); track all four camera-pan class probabilities.
(`expD7_composition.json`, script `steer_composition.py`.)
- **Superposition at both strengths.** s=100: joint steering gives P(right)=0.116 AND
  P(down)=0.080 simultaneously (21x and 26x baseline; singles 0.237 / 0.096). Each feature
  costs the other roughly half its solo effect, but both percepts survive — no
  winner-take-all, no interference.
- Selectivity is exact: P(left) and P(up) never move (0.003, at baseline) under any
  condition.
- Claim: two direction features steered together produce a right-AND-down mixture — the
  dictionary's direction code is approximately additive, as far as SSv2's class inventory
  can express it (no diagonal classes exist to read out directly).

### 32. Experiment D6 — cross-dictionary replication (BatchTopK SAE)
Design: trained a BatchTopK SAE (k=32, x8, 5000 steps, decay_start 4000) on the same cached
layer-11 train activations; dumped raw ball-token activations once
(`ball_raw_acts.pt`, 3000x8x768 — reusable asset: NFP on any dictionary is now a CPU job);
ran NFP through the new dictionary and compared with the main SAE's 85.
(`expD6_nfp_btk.json`, scripts `dump_ball_raw_acts.py`, `nfp_on_dict.py`.)
- **NFP-the-procedure replicates:** the BatchTopK dictionary yields 45 flagged features
  (0.73%), with all five taus represented (speed 7, vel_x 18, vel_y 11, accel 6, dir 3) —
  same order of magnitude and similar tau spread as the main SAE's 85 (1.38%).
- **Individual feature directions do NOT transfer:** per-feature best |cos| against the
  main 85 is median 0.31; only 5/45 above 0.5. Different SAE architectures factor the
  temporal subspace differently (consistent with the SAE literature).
- **The SPANS share a significant common core:** principal-angle analysis gives span
  overlap 0.336 vs 0.209 +/- 0.012 for random same-size spans (~10 sd above null), with
  12 principal directions at cos >= 0.7 and the top at 0.909. A shared ~dozen-dimensional
  temporal core is found by both dictionaries; the rest is dictionary-specific.
- Framing: the model-level claim (a temporal subspace exists and NFP finds it) is
  architecture-independent; the feature-level inventory is not. Matches C4's lesson that
  the span, not the individual features, is the right unit for necessity claims.

### 33. Experiment E — specificity battery over all positive results
User request: verify the positive effects hold only for the identified features, not any
features. New controls: two fresh random-85 seeds AND an activation-matched-85 set (per
NFP feature, the nearest non-flagged feature by mean activation over the 4,300-video
cache; matching verified: NFP 0.0826 vs matched 0.0825 vs random 0.0376 — random features
are indeed 2.2x less active, so the matched control closes a real fairness gap).
(`expE_specificity.json`, script `controls_specificity.py`.)
- C1 transplant (camera pairs): NFP +3.33/+2.16 dLO; all controls <= +0.10 incl. matched.
- C2 shuffle-restore: NFP 17%; random 1%/-0%; matched 6% (active features carry a little
  generic restorable signal; the NFP set is ~3x the matched control).
- C4 span erasure (11 family classes): baseline 0.764 -> NFP-span 0.118; random spans
  0.764 / 0.682; matched span 0.636. NFP destroys -0.65; strongest control -0.13.
- D2 repair (alpha=3, 59 re-mined errors): NFP 16.9%; controls 0% / 3.4% / 3.4% —
  activation level does not explain repair.
- D7 composition: random pairs do nothing (P ~ 0.002-0.006); real+random elevates only
  the real feature's class (P(right)=0.208, P(down)=0.0035); only the real pair gives
  both percepts. Composition is feature-specific.
- Verdict: every positive effect from the C and D series survives two fresh random seeds
  and the activation-matched control. The effects are properties of the identified
  temporal features, not of active or arbitrary features in general.

### 34. Experiment F2 — structured decorrelation design (new-theory section 4): all five
targets feasible with the existing profile family
Implemented the LP/QP stimulus design end-to-end (`design_decorrelated_stimulus.py`,
`expF2_decorr_design.json`): 4000 profiles sampled from the EXISTING NFP generator, tau at
the 8 tubelet steps, C[j,k] = Cov_t(confound_k, target_a) per profile, empirical Chebyshev
sign check per (target, confound), then solve for weights (w >= 0, sum 1, C^T w = 0,
weighted Var(target) >= 0.5 pool mean; linprog + spread cap w_j <= 3/M since cvxpy is not
installed locally).
- **All 5 targets solvable; no provably-impossible (single-signed) confound pair exists in
  our concept set.** The Chebyshev hazard the theory warns about does not bite for
  {speed, vel_x, vel_y, accel_mag, direction} under the existing profile families.
- Residual covariances at machine precision (1e-17 to 1e-9). Effective sample size
  1334/4000 (= M/cap, the spread cap binds). Weight compositions are interpretable:
  speed-target upweights family-B (fixed-speed turns: back_and_forth/gradual/sharp = 62%);
  vel_x-target upweights A:constant (20%); accel-target upweights family-B (49%).
- Two design notes for the regeneration: (a) the variance constraint binds exactly at its
  floor (weighted Var = 0.5 pool mean for every target) — decorrelation trades against
  target variance; raise var_frac if more signal is wanted and re-check feasibility;
  (b) skewed-sign confounds exist (Cov(direction, vel_y): 46% pos / 9% neg) but remain
  solvable.
- Deliverable: per-target profile spec (counts, family, type, speed, direction, delta)
  saved for regenerating a 3000-video v2 dataset per target; S1/S3 (uniform independent
  spawn) unchanged, so the NFP guarantee is preserved while the confound covariances are
  zeroed by construction.
- **Amendment (user question: does the current set already satisfy section 4?): NO.**
  Uniform-weight covariances of the current design (M=8000): all off-diagonal couplings
  ~0 EXCEPT **Cov(direction, vel_y) = +0.68** (t >> 4; variances ~1.0). Most pairs cancel
  by the original design's symmetries (uniform theta, sign-paired profiles: e.g.
  Cov(vel_x, speed) = cos(theta) Var(s), E[cos theta] = 0). The direction-vel_y coupling
  survives by PARITY: integral of theta*sin(theta) over the circle is positive (odd*odd =
  even integrand), while theta*cos(theta) integrates to 0 — no heading randomization can
  cancel it; only profile reweighting can (the LP uses the 9% negative-cov arcs that
  cross the +/-pi wrap). This single coupling explains F1's c_bar[direction] ~
  c_bar[vel_y] (cos 0.96), the direction->vel_y selectivity bleed, and plausibly the
  scarcity of direction-tagged features (2/85). Section 4's LP is needed for exactly this
  pair; the rest was already handled by the original design.

### 35. Experiment F1 — the max-covariance direction c_bar (new-theory section 2): detects,
steers only where detection overlaps the readout, and exposes a stimulus confound F2 fixes
Computed c_bar_tau for all 5 taus in closed form from ball_raw_acts.pt; steered with
delta * unit(c_bar) at layer 11 (delta +/-150), matched pairs, vs two random unit
directions and the head-diff direction (the theory's per-unit-norm ceiling).
(`steer_cbar_direction.py`, `expF1_cbar_steering.json`.)
- **Structure:** c_bar[vel_y] and c_bar[direction] are nearly the SAME direction
  (cos 0.96) — the ball stimulus cannot distinguish them, which matches F2's finding that
  Cov(direction, vel_y) is sign-skewed (46%/9%) in the current profile pool. The v2
  decorrelated design is exactly the fix. Max |cos| to any NFP decoder column: 0.25-0.50
  (c_bar is not any single feature).
- **Misalignment finding (theory note predicted this would be interesting):
  cos(c_bar, head-diff) <= 0.36 on every matched pair** — the model's detection-optimal
  temporal directions are NOT the directions its classification head reads for those
  discriminations.
- **Steering: works on camera pairs only, at intermediate strength.** cam_lr shift -3.80
  (flip 29%), cam_ud +3.87 (flip 29%) vs random ~0.4 — c_bar beats the best single NFP
  feature (~1.7) but is ~3x below the head-diff ceiling (11-13, 100% flips at the same
  unit norm). Object pairs and fall_speed/spin_stop: within random range. E[speed]
  dose-response on neutral videos: flat. Same scope boundary as all ball-derived objects.
- Verdict for the theory note's open question: c_bar is a detection direction; steering
  with it inherits the detection/steering split we measured at the feature level (step
  20). It steers only where the ball-visible subspace overlaps what the head reads
  (global/camera motion), at a fraction of the task-optimal direction's strength. The
  head-diff at unit norm flips every pair at 100% — the strongest steering ceiling
  measured in the project.

### 36. NFP v2 — decorrelated dataset built per new-theory section 4 (generation launched)
Pipeline (all new code compiled + smoke-tested):
- **Joint design solved**: one dataset, ALL 10 pairwise couplings zeroed simultaneously
  (`design_decorrelated_stimulus.py --joint`): residuals <= 3.4e-9, ESS 1335/4000,
  weighted Var comfortably above floors for all 5 taus. Exact-N (3000) allocation by
  largest remainder; per-video spec in `local_runs/steering/nfp_v2_profile_spec.json`
  (copy staged at `data/nfp_v2_profile_spec.json`). End-to-end verification through
  profile reconstruction: all 10 dataset-level covariances <= 0.002 (vs +0.68 for
  direction-vel_y in v1; residual = integer rounding).
- **Generator extended** (`data/nfp_ball_dataset.py`): `--profile_spec` + `profile_from_spec`
  rebuild vx/vy deterministically from saved parameters; start positions still drawn from
  the per-video RNG -> S1/S3 preserved AND positions are IDENTICAL to v1 (same seed+idx
  RNG), so v2 differs from v1 only in velocity profiles. New runner
  `data/run_nfp_v2_dataset.ps1`. Smoke test (2 videos): rendered metadata matches spec
  exactly.
- **Full render running**: 3 parallel Kubric-Docker shards (0-999 / 1000-1999 /
  2000-2999) -> `data/output/nfp_v2/`; resume-safe. Docker Desktop was started for this.
- Analysis plan when render completes: (1) verify dataset covariances from rendered
  metadata; (2) `dump_ball_raw_acts.py --dataset_dir data/output/nfp_v2` (GPU, ~30 min);
  (3) NFP test on v2 with the SAME SAE (trained on SSv2, so unchanged) -> new flags/tags,
  direction row of the selectivity matrix, direction-tagged feature count (v1: 2/85);
  (4) c_bar v2: check cos(c_bar[direction], c_bar[vel_y]) (v1: 0.96 -> expect ~0).

### 37. Experiment F3 — NFP on the v2 decorrelated dataset (same SAE, new stimulus)
Render verified: all 3000 videos; dataset covariances all <= 0.0022, direction-vel_y
+0.680 -> +0.00044 (1500x). Same SAE, same statistic, same bar.
(`nfp_v2_analysis.py`, `expF3_nfp_v2.json`, `ball_raw_acts_v2.pt`.)
- **P1 CONFIRMED decisively: cos(c_bar[direction], c_bar[vel_y]) = 0.96 (v1) -> -0.17
  (v2).** All v2 off-diagonal c_bar cosines <= 0.34. Stimulus decorrelation disentangles
  the recovered concept directions exactly as the additive-encoding theory predicts.
- **Flags: 85 -> 109 (1.77%), stable core of 70/85 re-flagged.** Speed-significant grew
  40 -> 66 (the joint design raised speed variance 0.40 -> 0.55, boosting speed power).
- **P3 nuanced: direction-SIGNIFICANT collapsed 15 -> 4** — v1's direction significances
  were mostly the vel_y coupling leaking through. v2's direction-dominant features are 3
  and are ALL NEW indices (feat00115, feat01217, feat03747): genuine direction detectors
  previously masked. The v1 direction tag was mostly artifact; v2's is honest and small.
- **P2 NOT confirmed, informatively: the direction (and vel_y, accel) selectivity rows
  now peak on speed.** Two causes: (a) higher speed variance in the design; (b) the
  irreducible ReLU entanglement — a half-wave-rectified velocity feature
  (f = ReLU(vel_y-ish)) covaries positively with speed within its preferred-direction
  videos EVEN when Cov(vel_y, speed) = 0 at the dataset level. Same mechanism as the
  synthetic control's swing-size leak and the theory PDF's Claim 2 / Chebyshev
  discussion: stimulus design can zero LINEAR couplings; it cannot make rectified
  features stop responding to speed. Post-decorrelation, the selectivity matrix now
  measures the FEATURES' true nonlinear response profile rather than stimulus confounds.
- Net: v2 upgrade validated. c_bar directions disentangled; direction tags now honest;
  the remaining speed cross-response is a property of the features, not the stimulus,
  and is itself evidence for the rectified-encoding picture.

### 38. Experiment F4/F1b — steering with v2 assets (new direction features + c_bar_v2)
(`expF4_v2dir_flip.json`, `expF1b_cbar_v2_steering.json`.)
- **New direction features on pairs:** feat01217 moves the object-level PUSHING pair —
  shift -3.04, pair-flips 4% -> 25% — the first probe-flagged feature to move that pair
  at all (every v1 NFP feature: <= 8%). Caveat: its +s attractor is sharp
  ("Pushing with something", p=0.72) and it also moves move_ud (25%), so this is partly
  attractor-flavored motion toward pushing classes, not a clean L/R knob. feat00115 and
  feat03747: weak (shifts <= 1.1, flips <= 12%).
- **c_bar_v2 steering:** cam_lr IMPROVED over v1 — shift -4.08 vs -3.80, flips 42% vs
  29%, top1 25% vs 12% (decontaminating vel_x sharpened its steering direction). cam_ud
  similar to v1 (+3.24). The disentangled c_bar[direction] STILL does not steer
  direction pairs (shift +1.48, 0 flips) and its cos to the head-diffs stays <= 0.27:
  detection-optimal != steering-optimal survives stimulus decorrelation, because the
  direction axis the head reads is carried by ball-OOD features regardless of probe
  quality. Object pairs, fall_speed, spin_stop, and the E[speed] dose-response remain at
  random/flat for all c_bar directions.
- Net: v2 sharpens what the probe can see (camera/global motion steering up ~1.4x on
  flips) and surfaces one genuinely new, partially-steering direction feature; it does
  not, and cannot, move the model's object-direction machinery into the probe's reach.

### 39. Experiment E2 — closing the last two control gaps
(`controls_direction_nulls.py`, `expE2_direction_nulls.json`.)
- **c_bar direction nulls:** 30-direction battery (20 uniform-random + 10 manifold-
  matched = random unit combos of the top-50 ball-activation PCs; manifold nulls are
  ~2x stronger than uniform, validating the concern). cam_lr: c_bar |shift| 4.08 vs
  null mean 0.60 / max 2.52, flips 0.42 vs null max 0.12, empirical p = 0/30. cam_ud:
  3.24 vs null max 2.90, p = 0/30. No null direction reaches the c_bar effect.
- **Reversal-restore extra controls:** two fresh random seeds recover 1% and 0%,
  activation-matched 2%, vs NFP-85 13%. Multi-seed + matched coverage now complete.
- With this, every item on the valid-evidence list (steps: pair screen/ExpA, C1, C2,
  D3, C4, D2, F1b) carries multi-seed and matched-strength ablations. The answer to
  "could random features or directions produce these effects" is measured: no.
- Consolidated document: `REPORT_EVIDENCE.md` (design/parameters/intended effect/
  observation/ablations per experiment, grouped by NFP version).

## FINAL VERDICT
Steering single temporal SAE features in VideoMAE works -- robustly and interpretably:
1. Population: significant features steer toward their own data-defined concept far more than
   random (MW p=2.5e-4 at s=100, 2.1e-3 at s=40); clamp-robust (per-feature z r=0.97 across s).
2. ~10 strong, motion-coherent features (vel_y->lifting, speed->spinning/rolling, vel_x->pushing).
3. Causal demo: on a MOTIONLESS frame, steering one feature injects its specific motion, with a
   clean monotone dose-response (feat02312 -> P(spinning-quickly-stops)=0.30; feat03714 -> 0.34).
4. Honest limits: (a) crude hand-built axes (force, motion-keyword) never separate from null --
   only the data-defined own-concept readout does; (b) a supervised diff-of-means direction
   steers a chosen axis ~7x harder than any single feature (AxBench) -- SAE features trade power
   for interpretability (85 pre-labeled directions, no supervision); (c) not every concept-align
   winner gives a clean static dose-response (2/5 flat).

It is NOT a coincidence: multiple features, two clamp strengths, a permutation null, a
random-feature null, and a frozen-frame causal demo all agree.

## Best demo features (motion-coherent, strong across tests)
- feat02312 [speed]  -> spinning-quickly-stops  (concept-align z=6.8; frozen-frame +0.18) STAR
- feat04280 [vel_y]  -> lifting / vertical       (concept-align z=11.3, strongest)
- feat03150 [speed]  -> rolling                  (concept-align z=7.4)
- feat02211 [speed*] -> barely-moves (fires-on-slow)  (concept-align z=6.7)
- feat03714 [vel_x]  -> pushing slightly         (concept-align z=8.5; frozen-frame coherent)

## Files
- analysis/steer_ssv2_logits.py      — pipeline + SteerLayer (clamp / record / add_vec)
- analysis/steer_feature_sweep.py    — broad force-axis + S_f sweep (step 2)
- analysis/steer_concept_alignment.py— LLaVA-faithful own-concept readout (step 3)
- analysis/steer_static_input.py     — frozen-frame motion injection
- analysis/steer_diffmeans.py        — linear diff-of-means ceiling
- local_runs/steering/expA_*.json    — all results
