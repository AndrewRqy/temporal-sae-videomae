"""
Objective per-class speed score for SSv2 classes, via optical flow.

For each of the 174 SSv2 classes, sample up to M validation videos, decode 16 uniformly-spaced
frames (the same temporal sampling VideoMAE sees), resize to 224, and measure the mean
Farneback optical-flow magnitude between consecutive frames. Average over frames and videos ->
a continuous "how much motion does this class contain" score per class.

This gives a data-defined speed axis over the 174 classes so that steering experiments can read
out E[speed | prediction] = sum_c P(c) * speed(c), and rank a feature's concept classes into a
highest-speed C_hi and lowest-speed C_lo for the bidirectional within-class steering test.

Output: local_runs/steering/class_speed.json
  { "by_idx": {c: {"label":..., "speed":..., "speed_z":..., "n":...}}, "mean":..., "std":... }

Usage (from sae-for-vlm/):
  python analysis/class_speed_scores.py --per_class 6
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from transformers import VideoMAEForVideoClassification

N_FRAMES = 16


def decode_frames(path, size=224):
    import av
    try:
        cont = av.open(str(path))
        frames = [f.to_ndarray(format="rgb24") for f in cont.decode(video=0)]
        cont.close()
        if len(frames) < 2:
            return None
        idx = np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)
        out = [cv2.resize(frames[j], (size, size)) for j in idx]
        return out
    except Exception:
        return None


def clip_flow_mag(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mags = []
    for a, b in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
    return float(np.mean(mags)) if mags else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--per_class", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out", default="local_runs/steering/class_speed.json")
    args = ap.parse_args()

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name)
    id2label = clf.config.id2label
    label2idx = {v: int(k) for k, v in id2label.items()}
    n_cls = len(id2label)

    val = json.load(open(args.ssv2_val_json))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(val)
    by_cls = {}
    for it in val:
        c = label2idx.get(it.get("template", ""), -1)
        if c < 0:
            continue
        by_cls.setdefault(c, []).append(it["id"])

    vdir = Path(args.ssv2_videos)
    by_idx = {}
    print(f"{len(by_cls)} classes have val videos | per_class={args.per_class}")
    for n, c in enumerate(sorted(by_cls)):
        ids = by_cls[c][: args.per_class]
        vals = []
        for vid in ids:
            fr = decode_frames(vdir / f"{vid}.webm")
            if fr is None:
                continue
            m = clip_flow_mag(fr)
            if m is not None:
                vals.append(m)
        if vals:
            clab = id2label.get(str(c), id2label.get(c))
            by_idx[c] = {"label": clab, "speed": float(np.mean(vals)), "n": len(vals)}
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(by_cls)} classes done")

    speeds = np.array([v["speed"] for v in by_idx.values()])
    mu, sd = float(speeds.mean()), float(speeds.std())
    for v in by_idx.values():
        v["speed_z"] = round((v["speed"] - mu) / (sd + 1e-9), 3)
        v["speed"] = round(v["speed"], 4)

    ranked = sorted(by_idx.items(), key=lambda kv: -kv[1]["speed"])
    print(f"\nmean flow={mu:.3f} std={sd:.3f}  ({len(by_idx)}/{n_cls} classes scored)")
    print("\nTOP 10 fastest classes:")
    for c, v in ranked[:10]:
        print(f"  {v['speed']:6.3f} (z{v['speed_z']:+.2f})  {v['label']}")
    print("\nBOTTOM 10 slowest classes:")
    for c, v in ranked[-10:]:
        print(f"  {v['speed']:6.3f} (z{v['speed_z']:+.2f})  {v['label']}")

    out = {"mean": mu, "std": sd, "per_class": args.per_class,
           "by_idx": {str(c): v for c, v in by_idx.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
