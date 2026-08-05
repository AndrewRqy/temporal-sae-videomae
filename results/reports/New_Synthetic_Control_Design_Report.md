# The new synthetic positive control: design, rationale, and NFP results

## Part 1. Design

### 1.1 What a positive control is for

The NFP test claims it can find the features in a video model that encode motion
(speed, velocity, acceleration, direction). On a real model there is no answer key.
If the test flags 85 features, we cannot know how many are right.

A positive control builds a situation where the answer key is known. We construct fake
"video representations" out of ingredients we choose, so we know exactly which directions
in the representation space carry motion information and which do not. We then train a
sparse autoencoder (SAE) on these representations and run the NFP test on it. The SAE is
never told anything about the ingredients. If the test flags features aligned with the
motion ingredient and ignores everything else, it works. Every mistake is visible because
we hold the answer key.

### 1.2 The three ingredients

Each synthetic representation describes one video V at one of 8 time steps t. It is a
768-dimensional vector built as a sum of three signals plus noise:

```
h(V,t) =  motion signal  +  frozen content  +  moving content  +  noise
```

1. **Motion signal** (5 directions, `W_tau`). We take the same 3,000 ball trajectories as
   the NFP stimulus videos, with scripted velocity profiles (accelerate, decelerate, turn,
   back-and-forth). At each time step we write the ball's current speed, velocity
   components, acceleration, and heading into 5 fixed directions of the space. This is
   the signal the test must find.

2. **Frozen content** (100 directions, `W_static`). Each video gets a random code that
   never changes across its 8 steps. This models scene identity: which objects, what
   background. Real models are dominated by this kind of content, so we give it 20x more
   directions than motion. The SAE has to spend most of its capacity here.

3. **Moving content** (50 directions, `W_pos`). This is the new ingredient. In a real
   video, content features are not frozen: as things move, the scene at each location
   changes, so content activations drift from frame to frame. A control whose content
   never moves makes the test's job artificially easy, because a constant has zero
   covariance with everything. We model moving content as functions of the ball's
   position. Each channel is a bump on the canvas (a Gaussian centered somewhere, with a
   random + or - sign). Its value at time t is the bump's height at the ball's current
   position. As the ball moves, these channels rise and fall. They encode where the ball
   is, not how it moves.

The three ingredient subspaces are mutually orthogonal. A feature's alignment with each
block can therefore be measured directly.

### 1.3 The trap: position is the integral of velocity

Suppose all videos shared one fixed landscape of bumps. Take a channel whose bump slopes
upward toward the right side of the canvas, and a video of a ball accelerating rightward.
As speed rises, the ball climbs the slope faster, so the channel's value and the rightward
velocity rise together. Because the landscape is the same for every video, this happens in
the same direction for every video. The NFP test asks: do this feature and this motion
variable move together consistently across thousands of videos? For a fixed landscape the
answer is yes. We measured it: a shared bank of channels produces false motion signal in
the position block with test statistics up to |t| = 20.7 (Fourier basis) and 14.1
(Gaussian bumps). Both are far past the significance bar. Averaging over ball directions
does not remove this, because the displacement-velocity relationship survives direction
reversal. No arrangement of a fixed landscape fixes it.

### 1.4 The fix: every video gets its own landscape

The zero-correlation requirement is a dataset-level statement, not a per-video one.
Within a single video, moving content does covary with motion. That is realistic and
acceptable. What must vanish is the consistent direction of that relationship across
videos. So each video draws its own random landscape: its own bump centers and its own
random signs. In one video the channel rises with rightward motion, in the next it falls.
Over 3,000 videos the relationship averages to zero.

The random sign is required for exactness. Bumps are all-positive hills, and near the
canvas edges random centers alone leave a small directional bias. Flipping the sign of the
whole landscape half the time cancels it. This is the same role the random phase plays for
cosine-based channels.

This per-video mode is the default. It also matches reality better than a shared
landscape: every SSv2 clip is a different scene, so each video having its own content
layout is the right model. It also gives the test a proper null. The position channels
have real within-video covariance with motion (up to 0.37 in channel units) whose sign is
random across videos. That is exactly the hypothesis the NFP t-test claims to reject.

We verify this rather than assume it. The generator prints an oracle leakage table on
every run: the NFP statistic applied to the position channels themselves. For the current
data, the 250 channel-by-variable tests have max |t| = 3.11 and none above the
significance bar. This matches pure noise.

An alternative mode, `universal_resid`, keeps one shared landscape and subtracts each
video's motion-explainable component from every channel. This gives an exact per-video
zero, but the resulting null is degenerate: every covariance is exactly zero, which is
easier than reality. Both modes and both channel bases (Gaussian, Fourier) are kept
runnable.

### 1.5 Pipeline

Generate 3,000 x 8 representations. Train a ReLU SAE on them (6,144 features, expansion
x8, same hyperparameters as the cluster runs), blind to all ingredients. Run the NFP test
on the SAE features. Score every flagged feature against the answer key using ProjFrac:
the fraction of a feature's encoder direction lying in each ingredient's subspace.

## Part 2. NFP results on the new control

### 2.1 Headline numbers

- **17 of 6,144 features flagged as temporal (0.28%).** Per motion variable (a feature
  can be flagged for several): speed 4, vel_x 10, vel_y 12, direction 12, accel_mag 5.
- **Flagged features align with the true motion subspace.** Flagged features put on
  average 25.2% of their encoder direction inside the 5-dimensional motion subspace.
  Unflagged features put 0.41% there. That is a 61x separation. A random direction would
  land at 5/768 = 0.65%, so flagged features sit at about 39x chance. By variable:
  speed-flagged features average 54% in the motion subspace, vel_x 39%, vel_y 36%,
  direction 36%.
- **Unflagged features behave as claimed.** Their mean covariance with the motion
  variables is about 4e-5, which is numerically zero. Frozen-content directions never
  trigger the test.
- **Selectivity.** The response matrix is diagonal-dominant for speed, vel_x, and vel_y:
  features flagged for a variable respond most strongly to that variable. There is one
  systematic exception, described below.

### 2.2 Red flags

**Red flag 1: five accel_mag-flagged features have near-zero motion-subspace alignment
(ProjFrac = 0.0002).** These are position-content features that the test flags anyway.
Why it happens: think of a content channel as a bumpy landscape the ball drives across.
When the ball moves fast, it covers more landscape per time step, so the channel's value
swings harder. An SAE feature is a ReLU. It only reports the positive part of its input,
so bigger swings produce bigger average activations. The feature's activation level
therefore rises during fast or accelerating segments of every video. This is a real,
consistent correlation with motion, produced by a feature that encodes only position.
No randomization can remove it: flipping a landscape's sign flips which way the channel
moves but not how much it swings. Swing size is sign-proof, so this leak survives the
per-video randomization in both channel bases (2 such features with cosine channels, 5
with bumps). The conclusion it forces: a covariance-based test certifies that a feature is
temporally informative, which is a larger category than encodes kinematics. The
contamination is measured at about 5 features out of 6,144, concentrated in accel_mag,
and each case carries near-zero motion-subspace alignment, so the answer key catches all
of them.

**Red flag 2: the direction row of the selectivity matrix is not diagonal-dominant.**
Direction-flagged features respond slightly more to vel_y (0.024) than to direction
itself (0.019). Why it happens: heading is a function of the velocity components (it is
the angle of the velocity vector), so any feature tracking direction must also covary
with vel_x and vel_y. The variables are entangled. The new effect-size selectivity metric
(mean covariance on standardized variables, replacing the old consistency-based |t|)
shows this instead of hiding it. This is a property of the chosen variable set, not a
test failure.

**Red flag 3: the flagged fraction (0.28%) is well below the real VideoMAE run (1.38%).**
Why it happens: per-video random content cannot be compressed into a per-video code the
way frozen content can, so the SAE spends more of its dictionary representing it. Fewer
features are left for the motion signal, which carries about 6x less power than the
content blocks by design. The control became harder and cleaner at the same time. The old
constant-static control matched the real percentage more closely, but partly for
artificial reasons. If matching the percentage matters, the calibration knobs are the
number of position channels, their spatial scale, and the noise level.

**A nuance, not a red flag: 7 of the 17 flagged features have more of their direction in
the position block than in the motion block.** These are mixed features. The SAE
entangles a real motion component with content in a single feature, because nothing
forces it to separate them. They are flagged because of their motion content; their
encoder direction also carries content. This is expected SAE behavior at this sparsity,
and the per-variable alignment numbers above already account for it.

### 2.3 Summary

On a control where content both dominates and moves, the NFP test finds a small,
motion-aligned feature set (61x separation from the unflagged population) with no false
positives from frozen or moving content in the linear sense. Its measured failure mode is
small and specific: a handful of accel_mag flags caused by nonlinear swing-size coupling,
which no linear decorrelation can prevent and which the ground-truth answer key exposes.
