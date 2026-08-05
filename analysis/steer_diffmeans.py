"""
Baseline / ceiling control — diff-of-means steering toward fast motion.

AxBench's lesson: a single SAE feature may steer worse than a trivial difference-of-means
direction. This builds the best LINEAR motion direction at layer 11 and steers with it, to
answer "is this classifier steerable toward fast/forceful classes AT ALL?" — the ceiling our
single-feature steering should be judged against.

  v = mean(layer-11 residual over FAST-class clips) - mean(over SLOW-class clips)   [768-d]
Then on a neutral test set: add alpha*v to the residual (all tokens) and measure the force_shift
toward FAST classes, sweeping alpha. Fast/slow clips are picked by ground-truth template (no
decode needed to know the label), so v is built from real fast vs slow videos.

Usage (from sae-for-vlm/):
  python analysis/steer_diffmeans.py --n_build 64 --n_test 160
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
from analysis.steer_ssv2_logits import SteerLayer, SSv2Frames, ssv2_collate, build_fast_slow


class ItemFrames(SSv2Frames):
    """SSv2Frames but over an explicit pre-selected list of val items."""
    def __init__(self, videos_dir, items):
        self.items = items
        self.dir = Path(videos_dir)


def probs_and_raw(clf, steer, cached, device, want_raw=False):
    steer.captured_raw = []
    steer.record_raw = want_raw
    outs = []
    for pv in cached:
        with torch.no_grad():
            outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
    steer.record_raw = False
    P = torch.cat(outs, 0).numpy()
    R = torch.cat(steer.captured_raw, 0).numpy() if want_raw else None
    return P, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_build", default=64, type=int, help="clips per side to build v")
    ap.add_argument("--n_test", default=160, type=int)
    ap.add_argument("--alphas", nargs="*", type=float, default=[1, 2, 4, 8])
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_diffmeans.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    fast, slow = build_fast_slow(id2label)
    fast_lab = {lab(i) for i in fast}; slow_lab = {lab(i) for i in slow}
    fast_i, slow_i = np.array(fast), np.array(slow)

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    val = json.load(open(args.ssv2_val_json))
    rng = np.random.RandomState(args.seed); rng.shuffle(val)
    fast_items = [v for v in val if v.get("template") in fast_lab][: args.n_build]
    slow_items = [v for v in val if v.get("template") in slow_lab][: args.n_build]
    used = {v["id"] for v in fast_items + slow_items}
    test_items = [v for v in val if v["id"] not in used][: args.n_test]
    print(f"build: {len(fast_items)} fast + {len(slow_items)} slow clips | test: {len(test_items)}")

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    steer.enabled = False
    _, Rf = probs_and_raw(clf, steer, cache_of(fast_items), device, want_raw=True)
    _, Rs = probs_and_raw(clf, steer, cache_of(slow_items), device, want_raw=True)
    v = torch.tensor(Rf.mean(0) - Rs.mean(0), dtype=torch.float32)        # [768]
    print(f"diff-of-means vector ||v||={v.norm():.3f}")

    test_cache = cache_of(test_items)
    base, _ = probs_and_raw(clf, steer, test_cache, device)
    bm = base.mean(0)

    def force_shift(P):
        dP = (P - base).mean(0)
        return float(dP[fast_i].sum() - dP[slow_i].sum()), dP

    print(f"\nbaseline test FAST mass={base[:, fast_i].sum(1).mean():.4f} "
          f"SLOW mass={base[:, slow_i].sum(1).mean():.4f}")
    results = {"alphas": [], "||v||": float(v.norm())}
    for a in args.alphas:
        steer.add_vec = a * v
        P, _ = probs_and_raw(clf, steer, test_cache, device)
        steer.add_vec = None
        fs, dP = force_shift(P)
        up = np.argsort(-dP)[:5]
        rec = {"alpha": a, "force_shift": round(fs, 5),
               "top_up": [(lab(i), round(float(dP[i]), 4)) for i in up]}
        results["alphas"].append(rec)
        print(f"  alpha={a:<4} force_shift={fs:+.5f}  up:{rec['top_up'][0]}  {rec['top_up'][1]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
