"""
Dose-response demo (LLaVA Fig.6 analog) for the best steering features.

For each demo feature, define its concept class = the most common SSv2 ground-truth class
among its top-activating clips, then sweep the clamp strength s and track the mean predicted
probability of that concept class. A monotone rise = "turn the knob up, the concept appears
more" -- exactly the single-neuron steering demo, here for a temporal motion concept.

Two input modes:
  --input ssv2    : real SSv2-val clips (effect on top of real content)
  --input static  : frozen frame repeated 16x (inject the concept into a motionless clip)

Usage (from sae-for-vlm/):
  python analysis/steer_dose_response.py --input static --features 2312 4280 3150 2211 3714
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
from analysis.steer_ssv2_logits import SteerLayer, SSv2Frames, ssv2_collate
from analysis.steer_static_input import StaticFrames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--features", nargs="*", type=int, default=[2312, 4280, 3150, 2211, 3714])
    ap.add_argument("--s_list", nargs="*", type=float, default=[0, 10, 20, 40, 60, 100, 150])
    ap.add_argument("--input", default="static", choices=["ssv2", "static"])
    ap.add_argument("--top_clips", default=25, type=int)
    ap.add_argument("--n_videos", default=256, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_dose_response.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    label2idx = {v: int(k) for k, v in id2label.items()}

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    # concept is always defined from REAL clips' activations (motion present); the demo can then
    # be shown on real or static inputs.
    real_ds = SSv2Frames(args.ssv2_videos, args.ssv2_val_json, args.n_videos, args.seed)
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                         collate_fn=ssv2_collate(proc))
    real_cache, gt = [], []
    for b in real_dl:
        real_cache.append(b[0]["pixel_values"]); gt += [label2idx.get(tm, -1) for tm in b[2]]
    gt = np.array(gt)

    steer.enabled = False; steer.record = True; steer.captured = []
    for pv in real_cache:
        with torch.no_grad():
            clf(pixel_values=pv.to(device))
    A = torch.cat(steer.captured, 0).numpy(); steer.record = False
    concept = {}
    for k in args.features:
        order = np.argsort(-A[:, k])
        top = [gt[j] for j in order[: args.top_clips] if gt[j] >= 0]
        cls = int(np.bincount(top).argmax()) if top else -1
        concept[k] = cls
        print(f"feat{k:05d} concept class = {lab(cls)}")

    # demo input set
    if args.input == "static":
        ds = StaticFrames(args.ssv2_videos, args.ssv2_val_json, args.n_videos, args.seed, "first")
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=ssv2_collate(proc))
        cache = [b[0]["pixel_values"] for b in dl]
    else:
        cache = real_cache

    def mean_probs():
        outs = []
        for pv in cache:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        return torch.cat(outs, 0).numpy().mean(0)

    print(f"\n=== dose-response ({args.input} inputs) : mean P(concept class) vs clamp s ===")
    results = {"input": args.input, "s_list": args.s_list, "features": []}
    for k in args.features:
        c = concept[k]
        row = {"feature": f"feat{k:05d}", "idx": k, "concept": lab(c), "p_by_s": []}
        ps = []
        for s in args.s_list:
            steer.enabled = (s != 0); steer.k = k; steer.s = s
            mp = mean_probs(); steer.enabled = False
            ps.append(float(mp[c]))
        row["p_by_s"] = [round(x, 4) for x in ps]
        results["features"].append(row)
        curve = "  ".join(f"s{int(s)}={p:.3f}" for s, p in zip(args.s_list, ps))
        print(f"  feat{k:05d} [{lab(c)[:34]:34s}] {curve}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
