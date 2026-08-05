# Steering temporal SAE features — proposal & research synthesis

> Local working note. NOT for git/HF yet (lives under the git-ignored `local_runs/`).
> Goal: an SAE-steering experiment for our VideoMAE temporal features, analogous to the
> LLaVA/CLIP single-neuron steering in Pach et al. (NeurIPS 2025, arXiv:2504.02821 —
> the paper this repo builds on).

## The reframe
The LLaVA demo steers CLIP because CLIP feeds an LLM. Our SAE is on VideoMAE
(`MCG-NJU/videomae-base-finetuned-ssv2`), which is **itself an action classifier with a
174-way output**. So VideoMAE's own logits are a steerable output — and reading them is
**more causally rigorous** than the LLaVA demo (same network, no second model, no
circularity with our detection test).

Premise is supported: "Interpreting Physics in Video World Models" (arXiv:2602.07050)
shows speed/acceleration are decodable AND steerable in VideoMAE-v2 / V-JEPA2.

## Candidate features (from steering_candidates.csv)
- **feat05087** — acceleration: |t| = [spd −0.1, vx −0.5, vy −0.3, **accel +8.5**, dir −1.4],
  selectivity 5.9× (the single most monosemantic temporal feature in the SAE).
- **feat02818** — speed/velocity-magnitude: |t| = [**spd −8.9**, vx −2.3, …], selectivity 3.8×.

## Proposals, ranked

### A — Steer feature, read VideoMAE's own SSv2 logits  (PRIMARY; building now)
Clamp feat05087 to value `s` on all 1568 tokens → re-add reconstruction error → finish the
forward pass → read 174-way logits. Hypothesis: mass shifts toward forceful/fast classes,
away from gentle/slight ones.
- SSv2 minimal pairs isolate kinematics from semantics:
  *poke so it slightly moves* → *poke so it falls over*; *push slightly* → *push off the table*;
  *throw* vs *pretend to throw*.
- **Money figure — double dissociation:** {accel feat, speed feat} × {impact/sudden-stop classes,
  sustained-motion classes}. Accel→hitting/"quickly stops spinning"/falls-off;
  speed→throwing/sustained motion.
- Uses the exact model the SAE was trained on; self-contained; runs on our GPU.

### B — VideoMAE→GPT-2 captioner  (optional eye-candy, mirrors paper visually)
`fztkm/lc_video_description_videomae_gpt2` (VideoMAE ViT-B + GPT-2, ActivityNet). Clamp the
feature in its encoder, check captions mention "fast/throw/accelerate" more.
**Caveat:** it uses a *Kinetics*-pretrained VideoMAE, NOT our SSv2 checkpoint → our SAE won't
transfer; we'd train a fresh SAE on its layer 11. Secondary demo only.

### C — Build a minimal video-MLLM  (most ambitious; park for now)
Frozen VideoMAE → SAE@L11 → projector → small LLM. No off-the-shelf video MLLM uses stock
VideoMAE (closest: VideoChat-InternVideo2, VideoMAEv2-derived). True LLaVA analog with text
output, but a project in itself.

## Steering recipe (residual stream, layer-11 post-MLP, 768-d)
`f = Enc(x); e = x − Dec(f); f_k := s (all tokens); x_out = Dec(f) + e`  → continue forward.
Equivalent to `x_out = x + n_k·(s − f_k)` where `n_k` = decoder column k. Re-adding `e` means
we edit only feature k and pay no reconstruction tax. VideoMAE has no CLS; head mean-pools
1568 tokens, so clamp all tokens. Calibrate `s` to a high percentile of the feature's own
activation, then sweep multiples (published clamp recipe; CLIP-ViT finds s≈150 saturates,
~10–15% of features steerable — arXiv:2504.08729; sae-for-vlm `attach_and_fix(neurons_to_fix={n:100})`).

## Controls / baselines (essential — AxBench arXiv:2501.17148 shows trivial baselines can beat SAEs)
- Random-feature clamp ×N → null distribution; target should sit in the tail.
- Non-temporal-feature clamp → specificity (shift is temporal, not generic perturbation).
- Magnitude-matched random direction → controls for "any perturbation of this norm".
- **Sign-flip**: clamp to 0/negative → opposite shift (toward gentle). Clean sign-flip = strong causal evidence.
- Diff-of-means baseline (mean fast-action − mean slow-action activations at L11).
- Reconstruction-error on/off → confirm effect is the edit, not reconstruction corruption.

## Metrics
- Force-shift score `Σ_{c∈Fast} Δp_c − Σ_{c∈Slow} Δp_c`, averaged over videos, with s-sweep curve (monotone→saturating).
- Per-class Δlogit on the minimal pairs.
- Steerability `S_f = mean_i (P̃_i − P_i)²` (CLIP-ViT metric) to rank our feature vs others.
- Bootstrap CIs over input videos.

## Input data note
No real SSv2 val clips locally (only the 3000 synthetic ball videos, which are OOD for the
SSv2 classifier). First pass validates the *mechanism* on ball videos (does clamping move
logits in the predicted direction, beat random features, sign-flip cleanly). The
publication-grade force-shift readout should use real SSv2-val clips — to obtain next.

## Positioning / gap
No prior work combines a sparse-feature SAE with explicit motion-concept steering on a video
model. Closest: Dokme & Vishwanath spatio-temporal SAEs on VideoMAE (arXiv:2604.03919; warns
vanilla TopK SAEs break temporal coherence — our features already pass the NFP test); motion
steering otherwise lives in VLA/robotics (Häon CoRL 2025; Swann SAE-VLA arXiv:2603.19183).

## Key references
- Pach et al., SAEs Learn Monosemantic Features in VLMs — arXiv:2504.02821 (this repo's basis)
- Steering CLIP's ViT with SAEs — arXiv:2504.08729
- Improving Steering Vectors by Targeting SAE Features — arXiv:2411.02193 (clamp vs add, error handling)
- AxBench — arXiv:2501.17148 (baselines)
- Interpreting Physics in Video World Models — arXiv:2602.07050 (speed/accel/direction in VideoMAE-v2/V-JEPA2)
- Spatio-Temporal SAEs — arXiv:2604.03919
- VideoMAE+GPT-2 captioner — hf.co/fztkm/lc_video_description_videomae_gpt2
