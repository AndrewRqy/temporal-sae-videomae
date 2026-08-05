"""
Experiment D1b — playback-speed mediation on the speed-defined class pair.

D1 showed generic SSv2 class predictions are largely invariant to 2x playback (action
identity does not depend on speed), so mediation had nothing to mediate. This version
uses the one class pair whose IDENTITY is a speed judgment:
  "[Something] falling like a feather or paper"  (slow fall)
  "[Something] falling like a rock"              (fast fall)

Speeding up a feather video makes the fall look rock-like. Three measurements:
  (a) Total effect: LO = log P(rock) - log P(feather) on feather videos at 1x vs 2x
      playback (2x = full-span sampling, 1x = middle-half, as in D1). TE = LO(2x) - LO(1x).
  (b) Mediation: run the 1x input, patch the chosen features' activations from the 2x
      run. NIE = LO(patched) - LO(1x); fraction = NIE / TE.
      Sets: NFP-85, speed-tagged subset, flippers-12, random-85, static-85, ALL-6144.
  (c) Control direction: rock videos at 2x (already fast; expected smaller TE).

Replication: seed 0; up to 30 feather and 30 rock videos with >= 40 frames from the
seed-1-shuffled validation list; model/SAE/flags as previous experiments.

Usage (from sae-for-vlm/):
  python analysis/steer_speed_mediation_pair.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from analysis.steer_ssv2_logits import SteerLayer
from analysis.steer_speed_mediation import TwoSpeedFrames, collate_two

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
FEATHER = "[Something] falling like a feather or paper"
ROCK = "[Something] falling like a rock"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_per_class", default=30, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD1b_speed_pair_mediation.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    c_feather, c_rock = label2idx[FEATHER], label2idx[ROCK]
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy(); t_all = nfp["t_stat"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = [int(i) for i in np.where((p_all < bonf).any(1))[0]]
    dom = {i: TAU[int(np.argmax(np.abs(t_all[i])))] for i in sig}
    speed_sub = sorted(k for k in sig if dom[k] == "speed")
    finite = np.isfinite(t_all).all(1)
    low_t = (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    screen = json.load(open(args.screen_json))
    flippers = sorted(r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1)
    rng = np.random.RandomState(args.seed + 7)
    sets = [("NFP-85", torch.tensor(sorted(sig))),
            (f"speed-{len(speed_sub)}", torch.tensor(speed_sub)),
            ("flippers-12", torch.tensor(flippers)),
            ("random-85", torch.tensor(sorted(rng.choice(
                [k for k in range(sae.dict_size) if k not in set(sig)], 85, replace=False)))),
            ("static-85", torch.tensor(sorted(rng.choice(static_pool, 85, replace=False)))),
            ("ALL-6144", torch.arange(sae.dict_size))]

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by = {FEATHER: [], ROCK: []}
    for it in val:
        t = it.get("template", "")
        if t in by and len(by[t]) < args.n_per_class * 2:
            by[t].append(it)

    def lo(P):
        return np.log(P[:, c_rock] + 1e-12) - np.log(P[:, c_feather] + 1e-12)

    results = {"sets": {}, "sides": {}}
    for side, items in [("feather", by[FEATHER]), ("rock", by[ROCK])]:
        ds = TwoSpeedFrames(args.ssv2_videos, items)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_two(proc))
        LO = {"1x": [], "2x": [], **{n: [] for n, _ in sets}}
        n_done = 0
        for pv1, pv2, ids in dl:
            if n_done >= args.n_per_class:
                break
            steer.record_tokens_idx = torch.arange(sae.dict_size); steer.captured_tokens = []
            with torch.no_grad():
                P2 = torch.softmax(clf(pixel_values=pv2.to(device)).logits, -1).cpu().numpy()
            f2 = steer.captured_tokens[0]; steer.record_tokens_idx = None
            with torch.no_grad():
                P1 = torch.softmax(clf(pixel_values=pv1.to(device)).logits, -1).cpu().numpy()
            LO["1x"] += lo(P1).tolist(); LO["2x"] += lo(P2).tolist()
            for name, idx in sets:
                steer.patch_idx = idx
                steer.patch_vals = f2[..., idx] if idx.numel() < sae.dict_size else f2
                with torch.no_grad():
                    Pm = torch.softmax(clf(pixel_values=pv1.to(device)).logits, -1).cpu().numpy()
                steer.patch_idx = steer.patch_vals = None
                LO[name] += lo(Pm).tolist()
            del f2
            n_done += pv1.shape[0]
        l1, l2 = np.array(LO["1x"]), np.array(LO["2x"])
        TE = l2 - l1
        print(f"\n=== {side} videos (n={len(l1)}): LO(rock-feather) 1x={l1.mean():+.2f} "
              f"2x={l2.mean():+.2f}  TE={TE.mean():+.3f} (se {TE.std()/np.sqrt(len(TE)):.3f}) ===")
        results["sides"][side] = {"n": len(l1), "LO_1x": round(float(l1.mean()), 3),
                                  "LO_2x": round(float(l2.mean()), 3),
                                  "TE": round(float(TE.mean()), 3)}
        for name, _ in sets:
            NIE = np.array(LO[name]) - l1
            frac = NIE.mean() / (TE.mean() + 1e-9)
            results["sides"][side][name] = {"NIE": round(float(NIE.mean()), 3),
                                            "frac": round(float(frac), 3)}
            print(f"  {name:<12} NIE={NIE.mean():+.3f}  fraction mediated={100*frac:.0f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
