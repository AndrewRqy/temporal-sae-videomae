# Proposal V2: remaining experiment directions for showing the identified features
# matter for temporal understanding

Status of evidence so far: the 85 NFP features are sufficient (steering, transplant,
restoration) and necessary (span erasure) for a narrow concept family (camera motion,
slight motion, spinning, vertical manipulation), and unimportant for global SSv2
performance (3% global restoration). The directions below either strengthen this claim,
extend it to new evidence types, or probe its boundaries. Ordered by expected value.

## D1. Playback-speed mediation (input-level temporal counterfactual)

Literature: speed perception and time-flow modeling in video (Seeing Fast and Slow,
arXiv 2604.21931); causal mediation analysis in networks (Vig et al. 2020; path
patching, arXiv 2304.05969; multi-mediator caveats, arXiv 2606.27510).

Idea: change the input's temporal statistics directly, with identical content, by
resampling frames. Speed up = sample 16 frames at 2x stride; slow down = sample at
half stride. This is a natural temporal intervention: no class pairs, no synthetic
stimuli, works on any video.

Design:
1. Measurement: for ~200 videos, compute the 85 features' activations at 0.5x, 1x, 2x
   playback. Prediction: speed-tagged features shift monotonically with playback rate;
   static features do not.
2. Mediation: the model's prediction changes between 1x and 2x versions (e.g. mass
   moves between slow-class and fast-class interpretations). Patch the 85 features'
   activations from the 2x run into the 1x run. Fraction mediated = (prediction shift
   under patch) / (total 1x-to-2x shift). Controls: random-85, static-85, all-6144
   ceiling. Readout: E[speed] over classes, or log-odds on speed-sensitive class sets.
Result that would matter: "changing input playback speed changes the prediction, and
X% of that change flows through the 85 identified features." A direct, quantified
mediation claim about temporal understanding, with content held exactly constant.
Cost: small. Frame resampling is free; ~4 passes per video per condition.

## D2. Error repair by feature amplification (positive performance intervention)

Literature: model editing by targeted component intervention (Editing predictions by
modeling model computation, gradientscience.org; SALVE, arXiv 2512.15938).

Idea: C4 showed removing the features' directions destroys their concept classes. The
positive converse: on videos of those classes that the model currently gets WRONG,
amplify the relevant features moderately (scale f_k by alpha in 1.5-3, staying near
natural range) and measure the repair rate.

Design: collect all validation errors on the 8 concept-family classes (camera pans,
lifting-one-end, tilting, poking-barely, pushing-slightly, spinning-continues).
Conditions: amplify the tag-matched NFP features; amplify random-85; amplify
static-85; no-op. Metrics: fraction of errors corrected to the true class; collateral
damage on currently-correct videos of other classes.
Result that would matter: "amplifying the identified features repairs N% of the
model's errors on temporal classes at no cost elsewhere." Importance stated as a
performance improvement, not a destruction.
Cost: small. Error mining is one clean pass over ~10-20 videos/class; repair passes
are a handful of conditions.

## D3. Reverse-play mediation (arrow-of-time test)

Literature: arrow-of-time recognition in video; direction-flip corruption preserves
appearance and speed exactly, unlike shuffling which destroys all order.

Idea: play videos backward. Direction percepts invert (left-right pairs swap, up-down
swaps, rising becomes falling); appearance and speed statistics are identical. Then
patch forward-play feature activations into the backward run.

Design: for the direction pairs, measure (a) how much reversal moves the prediction
toward the opposite pair class (it should, if the model reads direction), (b) whether
patching the 85 features' forward activations into the reversed run restores the
forward percept, vs controls. Also record per-feature activation changes under
reversal: vel-tagged features should change most; speed and static features should
not (a signature the tags predict).
Result that would matter: a second, cleaner corruption axis than shuffling (only
direction changes) where restoration through the identified features recovers the
percept. Complements C2 with a corruption that isolates exactly one temporal property.
Cost: trivial (frame order reversal) plus the standard patch passes.

## D4. SAE-bottleneck probe (sufficiency as a representation)

Literature: sparse probing and its baselines (Are SAEs useful? arXiv 2502.16681;
DeepMind negative results on SAE probing); concept-bottleneck models (Concepts in
Motion, arXiv 2509.20899).

Idea: train a linear classifier that sees ONLY the 85 features' mean-pooled
activations and measure what it can and cannot classify.

Design: logistic probes on frozen activations, 174-way and concept-family-only.
Feature sets: NFP-85, random-85, static-85, full 768 residual (ceiling), all 6144.
Report accuracy on the concept-family classes vs all other classes, plus a few-shot
curve (10/50/200 examples per class).
Result that would matter: "an interpretable 85-dimensional bottleneck of identified
features supports near-ceiling recognition of the temporal concept classes; matched
random features do not." Caveat to state up front: linear probes on raw activations
are a strong baseline (per the literature), so the claim is about the identified
subset's sufficiency and interpretability, not probe superiority.
Cost: small; activations can be cached in one pass, probes train on CPU.

## D5. Full-dictionary causal scan via attribution patching (NFP recall, measured)

Literature: attribution patching / AtP* (arXiv 2403.00745); gradient-based SAE
attribution (arXiv 2505.08080).

Idea: everything so far tests the 85 features NFP flagged. The complementary question:
how many causally-temporal features did NFP miss? Attribution patching approximates
every feature's patching effect with two forward passes and one backward pass, so all
6,144 features can be ranked cheaply for their causal effect on temporal-class logits
(e.g. on the pair log-odds used throughout).

Design: for each temporal pair, compute attribution scores for all features; take the
top-k causal set; measure overlap with (a) the NFP-85, (b) the axis features from
step 14. Validate the approximation on the 85 with the exact patching numbers we
already have.
Result that would matter: a measured precision/recall statement for NFP against a
causal ground truth: "NFP recovered X% of the model's causally-temporal features; the
missed ones are concentrated in ball-OOD real-scene features." Turns the step-14
anecdote into a number.
Cost: moderate; requires adding a backward pass through the SAE encode at layer 11.

## D6. Cross-dictionary replication

Literature: SAE features depend on architecture and training choices (multiple 2025
comparisons); our own repo has BatchTopK and Matryoshka checkpoints.

Idea: rerun NFP plus the span-erasure and restoration tests on a second SAE
architecture trained on the same activations. If the same concept classes are
destroyed and restored through the second dictionary's flagged features, the claims
are about the model's representation, not one SAE seed.

Design: NFP test on the BatchTopK SAE (checkpoint exists); intersect/compare flagged
sets; repeat C2 (restoration) and C4 (span erasure) with its flagged features.
Result that would matter: architecture-independence of the whole pipeline.
Cost: moderate (one NFP run plus two intervention runs).

## D7. Feature-code compositionality (lower priority)

Idea: steer a camera-right feature and a camera-up feature simultaneously and test
whether the prediction reflects composed motion. Limitation found in advance: SSv2 has
no diagonal-motion classes, so the readout must be indirect (relative mass on the two
component classes). Listed for completeness; weakest expected evidence per unit
effort.

## Recommended order

D1 (playback mediation) and D2 (error repair) first: both produce single-sentence
results about temporal understanding and model performance, both cheap, both use only
the identified features plus matched controls. D3 next (cheapest new corruption axis).
D4 and D5 are the quantitative wrap-ups (sufficiency-as-representation; NFP recall).
D6 when cluster time is available. D7 only if time remains.
