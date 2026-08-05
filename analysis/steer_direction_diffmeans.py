"""
Experiment B control — supervised ceiling for direction-pair flipping.

The single-feature flip test (steer_direction_flip.py) needs a ceiling: can ANY uniform
layer-11 residual direction flip "pushing left to right" <-> "right to left"? Build the
best linear direction for each pair (diff-of-means over real videos of the two classes)
and steer with it. If even this supervised vector cannot flip the pair, the failure of
single SAE features is a property of the layer/uniform-token intervention, not of the
features; if it can, the gap is attributable to the features themselves.

  v = mean(layer-11 residual | pos-class videos) - mean(| neg-class videos)   [768]
  steer: h <- h + alpha*v on all tokens, on held-out videos of both classes.

Usage (from sae-for-vlm/):
  python analysis/steer_direction_diffmeans.py --n_build 24 --n_test 24
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
from analysis.steer_direction_flip import PAIRS, ItemFrames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--alphas", nargs="*", type=float, default=[-8, -4, -2, 0, 2, 4, 8])
    ap.add_argument("--n_build", default=24, type=int)
    ap.add_argument("--n_test", default=24, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expB_direction_diffmeans.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    label2idx = {v: int(k) for k, v in id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    def probs_raw(cache, want_raw=False):
        steer.captured_raw = []; steer.record_raw = want_raw
        outs = []
        for pv in cache:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        steer.record_raw = False
        P = torch.cat(outs, 0).numpy()
        R = torch.cat(steer.captured_raw, 0).numpy() if want_raw else None
        return P, R

    results = {"alphas": args.alphas, "pairs": []}
    for pair in PAIRS:
        pos_l, neg_l = pair["pos"], pair["neg"]
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        pool_p, pool_n = by_tmpl.get(pos_l, []), by_tmpl.get(neg_l, [])
        if cp < 0 or len(pool_p) < args.n_build + 4 or len(pool_n) < args.n_build + 4:
            continue
        print(f"\n=== PAIR  {pos_l}  vs  {neg_l} ===")
        build_p, test_p = pool_p[: args.n_build], pool_p[args.n_build: args.n_build + args.n_test]
        build_n, test_n = pool_n[: args.n_build], pool_n[args.n_build: args.n_build + args.n_test]

        steer.enabled = False
        _, Rp = probs_raw(cache_of(build_p), want_raw=True)
        _, Rn = probs_raw(cache_of(build_n), want_raw=True)
        v = torch.tensor(Rp.mean(0) - Rn.mean(0), dtype=torch.float32)
        print(f"  ||v|| = {v.norm():.3f}  (diff-of-means, {len(build_p)}+{len(build_n)} build clips)")

        caches = {"pos": cache_of(test_p), "neg": cache_of(test_n)}
        rec = {"pos": pos_l, "neg": neg_l, "v_norm": float(v.norm()), "curves": {}}
        for side, src_cls, tgt_cls in [("pos", cp, cn), ("neg", cn, cp)]:
            lo_curve, flip_to_tgt = [], []
            for a in args.alphas:
                steer.add_vec = (a * v) if a != 0 else None
                P, _ = probs_raw(caches[side])
                steer.add_vec = None
                lo = float(np.mean(np.log(P[:, cp] + 1e-12) - np.log(P[:, cn] + 1e-12)))
                # flipping means: pos-side needs alpha<0 to go to neg; neg-side alpha>0
                fl = float((P[:, tgt_cls] > P[:, src_cls]).mean())
                t1 = float((P.argmax(1) == tgt_cls).mean())
                lo_curve.append(round(lo, 3)); flip_to_tgt.append((round(fl, 3), round(t1, 3)))
            rec["curves"][side] = {"pair_logodds": lo_curve, "flip(pair,top1)": flip_to_tgt}
            curve = "  ".join(f"a{int(a)}={x:+.2f}" for a, x in zip(args.alphas, lo_curve))
            print(f"  {side}-side LO(alpha): {curve}")
            best = max((f for f, _ in flip_to_tgt))
            print(f"  {side}-side flip-to-target across alphas (pair): "
                  + "  ".join(f"{f:.2f}/{t:.2f}" for f, t in flip_to_tgt) + f"   best={best:.2f}")
        results["pairs"].append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
