"""
Experiment D7 — compositionality of the direction code.

Do two direction features compose when steered together? SSv2 has no diagonal-motion
classes, so the readout is the joint behavior of the two component classes: steer a
camera-right feature and a camera-down feature separately and jointly on neutral videos
and track P(camera right), P(camera down), P(camera left), P(camera up).

Features (chosen by their step-19 attractor distributions, computed from weights):
  feat01321: +s attractor = "Turning the camera right"      (p=0.97)
  feat04665: +s attractor = "Turning the camera downwards"  (p=0.81)

Outcomes of interest at each clamp strength s in {40, 100}:
  - superposition: joint steering keeps BOTH class probabilities elevated
    (min(P_joint(right), P_joint(down)) >> baseline);
  - winner-take-all: one class absorbs the mass;
  - interference: both drop relative to single steering.
Also a budget-matched variant: joint at s vs singles at s (same per-feature strength)
and joint at s/2 (same total injected norm, approximately).

Inputs: 24 neutral videos (random classes, camera classes excluded), seed 0.

Usage (from sae-for-vlm/):
  python analysis/steer_composition.py
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
from analysis.steer_ssv2_logits import SteerLayer, ssv2_collate
from analysis.steer_pair_screen import ItemFrames

F_RIGHT, F_DOWN = 1321, 4665
CLS = {"right": "Turning the camera right while filming [something]",
       "left":  "Turning the camera left while filming [something]",
       "up":    "Turning the camera upwards while filming [something]",
       "down":  "Turning the camera downwards while filming [something]"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_videos", default=24, type=int)
    ap.add_argument("--s_list", nargs="*", type=float, default=[40, 100])
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD7_composition.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    cids = {k: label2idx[v] for k, v in CLS.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    cam_labels = set(CLS.values())
    items = [it for it in val if it.get("template", "") not in cam_labels][: args.n_videos]
    dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
    cache = [b[0]["pixel_values"] for b in dl]

    def mean_probs():
        outs = []
        for pv in cache:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        return torch.cat(outs, 0).numpy().mean(0)

    def run(ks, s):
        if ks is None:
            steer.enabled = False
        else:
            steer.enabled = True; steer.k = ks; steer.s = s
        p = mean_probs(); steer.enabled = False
        return {name: round(float(p[c]), 4) for name, c in cids.items()}

    res = {"features": {"right": F_RIGHT, "down": F_DOWN}, "conditions": {}}
    base = run(None, 0)
    res["conditions"]["baseline"] = base
    print(f"baseline: {base}")
    for s in args.s_list:
        for name, ks in [("right-only", [F_RIGHT]), ("down-only", [F_DOWN]),
                         ("joint", [F_RIGHT, F_DOWN])]:
            r = run(ks, s)
            res["conditions"][f"{name}@s{int(s)}"] = r
            print(f"  {name:<11} s={int(s):<4} " +
                  "  ".join(f"P({k})={v:.3f}" for k, v in r.items()))
        half = run([F_RIGHT, F_DOWN], s / 2)
        res["conditions"][f"joint@s{int(s/2)} (budget-matched)"] = half
        print(f"  joint-half  s={int(s/2):<4} " +
              "  ".join(f"P({k})={v:.3f}" for k, v in half.items()))

    # composition verdict at each s
    for s in args.s_list:
        j = res["conditions"][f"joint@s{int(s)}"]
        ro = res["conditions"][f"right-only@s{int(s)}"]
        do = res["conditions"][f"down-only@s{int(s)}"]
        verdict = ("superposition" if j["right"] > base["right"] * 3 and j["down"] > base["down"] * 3
                   else "winner-take-all" if max(j["right"], j["down"]) > 3 * min(j["right"], j["down"])
                   else "interference")
        res[f"verdict@s{int(s)}"] = verdict
        print(f"\n  s={int(s)}: joint P(right)={j['right']:.3f} P(down)={j['down']:.3f} "
              f"(singles: {ro['right']:.3f} / {do['down']:.3f}) -> {verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
