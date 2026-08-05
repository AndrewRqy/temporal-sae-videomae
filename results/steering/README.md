# Steering and causal experiment results

One folder per experiment; raw result files in each folder's `data/`, figures in
`figures/`. Scripts live in `analysis/`; consolidated write-ups in
`results/reports/` (Temporal_Feature_Evidence_Report.md, Steering_Experiments_Report.md,
NFP_v2_Construction_Report.md). `notes/` holds the chronological lab log (FINDINGS.md,
39 entries) and the two experiment proposals.

| folder | experiment | main scripts |
|---|---|---|
| A1_own_concept_steering | Clamp steering toward each feature's data-defined concept: population test vs random features, single-top-class variant, frozen-frame injection, dose-response, diff-of-means ceiling, per-feature scorecard | steer_ssv2_logits.py, steer_concept_alignment.py, steer_topclass.py, steer_static_input.py, steer_dose_response.py, steer_diffmeans.py, steer_feature_sweep.py |
| A2_speed_axis_and_generalization | Bidirectional 4-cell speed-axis test (9/85 motion knobs) and their generalization to a 12-class unseen panel; optical-flow class speed scores | steer_speed_axis.py, steer_generalize.py, class_speed_scores.py |
| B1_direction_flips | Direction-pair flipping: NFP features, supervised diff-of-means ceiling (100%), axis decomposition onto decoder columns, flips with the axis-carrying features | steer_direction_flip.py, steer_direction_diffmeans.py, decompose_direction_axis.py |
| B2_pair_screen | Unselected screen: all 85 features x 10 temporal pairs, both clamp signs; static-feature control screen; zero-ablation and hard-video necessity tests | steer_pair_screen.py, steer_pair_controls.py, ablation_hard_videos.py |
| C1_transplant | Interchange intervention: donor-to-receiver patching of natural activations between opposite-direction videos | steer_transplant.py |
| C2_shuffle_restore | Frame-shuffle corruption + restoration of feature subsets (recovery fractions) | steer_shuffle_restore.py |
| C3_global_restore | Same restoration at whole-task scale (all 174 classes, top-1 accuracy): the honest global null | steer_global_restore.py |
| C4_span_erasure | Necessity: erase the span of the 85 decoder columns; class-selective destruction of the 11 family classes | steer_span_erasure.py |
| D1_speed_mediation | Playback-speed counterfactual + mediation (null: model is speed-invariant at 2x; includes the rock/feather variant) | steer_speed_mediation.py, steer_speed_mediation_pair.py |
| D2_error_repair | Amplifying features repairs model errors on family classes (16.9%, dose-monotone, no collateral) | steer_error_repair.py |
| D3_reverse_play | Time-reversal corruption + restoration; tag physics signature (accel features invariant under reversal) | steer_reverse_play.py |
| D4_bottleneck_probe | Linear probes on frozen feature subsets, 4,300 videos (family-class sufficiency, few-shot curves). The 116 MB activation cache (expD4_probe_cache.pt) is excluded from git; regenerate with probe_bottleneck.py --phase cache | probe_bottleneck.py |
| D5_attribution_scan | Full-dictionary attribution patching (group-level convergence with the axis decomposition; per-feature validation failed, use as screener only) | attr_temporal_scan.py |
| D6_cross_dictionary | NFP through a BatchTopK SAE trained on the same activations: procedure replicates, feature identities do not, spans share a common core | nfp_on_dict.py, dump_ball_raw_acts.py |
| D7_composition | Joint steering of two direction features: superposition with exact class selectivity | steer_composition.py |
| E_specificity_controls | The specificity batteries: two fresh random seeds + activation-matched features for every positive result; 30-direction null battery (20 random + 10 manifold-matched) | controls_specificity.py, controls_direction_nulls.py |
| F_theory_cbar_and_v2 | The theory-note program: c_bar max-covariance directions (v1 and v2), the joint-LP decorrelated stimulus design, NFP v1-vs-v2 comparison, v2 direction-feature steering | steer_cbar_direction.py, design_decorrelated_stimulus.py, nfp_v2_analysis.py, steer_direction_flip.py |
| attractor_io_analysis | Detection-strength vs steering-power independence (input/output scatter figure); attractor spectra were computed from weights inline (FINDINGS steps 19-20) | - |

Large artifacts intentionally not in git (available or reproducible): the NFP v2 video
set, its raw ball-token activations, and the synthetic-control package are on the
HuggingFace dataset `AndrewRqy/temporal-sae-videomae` (folders `nfp_v2/`,
`synth_control/`); the D4 probe cache and SAE checkpoints are regenerable from scripts.
