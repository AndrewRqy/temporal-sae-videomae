"""
Experiment D2 — error repair by feature amplification (positive performance intervention).

C4 showed erasing the NFP features' span destroys their concept classes. The positive
converse: on videos of those classes that the model currently gets WRONG, amplify the
features moderately and measure the repair rate.

Concept-family classes: loaded from expC4_span_erasure.json, classes whose NFP-span
erasure damage exceeds random-span damage by >= 0.4 (the empirically feature-dependent
classes; ~8-12 classes).

Amplification: f_k <- alpha * f_k on all tokens for the chosen feature set (capture the
features' own activations, patch back scaled; decode + re-add error). This scales each
feature within its natural activation pattern; no fixed clamp, so inactive features stay
inactive and spatial structure is preserved.

Conditions: sets {NFP-85, flippers-12, random-85, static-85} x alpha {1.5, 2, 3}.
Metrics:
  - repair rate: fraction of previously-wrong videos whose top-1 becomes the true class;
  - collateral: accuracy change on (a) previously-correct videos of the same classes,
    (b) a 200-video sample of other classes, under the best NFP condition.

Replication: seed 0; error mining over up to 24 videos per family class from the
seed-1-shuffled validation list; model/SAE/flags as in all D experiments.

Usage (from sae-for-vlm/):
  python analysis/steer_error_repair.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--c4_json", default="local_runs/steering/expC4_span_erasure.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--excess_bar", default=0.4, type=float)
    ap.add_argument("--mine_per_class", default=24, type=int)
    ap.add_argument("--alphas", nargs="*", type=float, default=[1.5, 2.0, 3.0])
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD2_error_repair.json")
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

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy(); t_all = nfp["t_stat"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = [int(i) for i in np.where((p_all < bonf).any(1))[0]]
    finite = np.isfinite(t_all).all(1)
    low_t = (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    screen = json.load(open(args.screen_json))
    flippers = [r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1]
    rng = np.random.RandomState(args.seed + 7)
    sets = [("NFP-85", torch.tensor(sorted(sig))),
            ("flippers-12", torch.tensor(sorted(flippers))),
            ("random-85", torch.tensor(sorted(rng.choice(
                [k for k in range(sae.dict_size) if k not in set(sig)], 85, replace=False)))),
            ("static-85", torch.tensor(sorted(rng.choice(static_pool, 85, replace=False))))]

    c4 = json.load(open(args.c4_json))
    fam = [r["cls"] for r in c4["per_class_top"] if r["excess"] >= args.excess_bar]
    print(f"concept-family classes (C4 excess >= {args.excess_bar}): {len(fam)}")
    for c in fam:
        print(f"   {lab(c)}")

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_cls = {}
    for it in val:
        c = label2idx.get(it.get("template", ""), -1)
        if c >= 0:
            by_cls.setdefault(c, []).append(it)

    def run(cache, patch_sets=None):
        """patch_sets: (idx, alpha) -> capture own features, patch scaled."""
        preds = []
        for pv in cache:
            pv = pv.to(device)
            if patch_sets is not None:
                idx, alpha = patch_sets
                steer.record_tokens_idx = idx; steer.captured_tokens = []
                with torch.no_grad():
                    clf(pixel_values=pv)
                steer.record_tokens_idx = None
                fvals = steer.captured_tokens[0]
                steer.patch_idx = idx; steer.patch_vals = alpha * fvals
            with torch.no_grad():
                preds.append(clf(pixel_values=pv).logits.argmax(-1).cpu().numpy())
            steer.patch_idx = steer.patch_vals = None
        return np.concatenate(preds)

    # --- mine errors and correct videos on family classes ---
    wrong_items, wrong_lbls, right_items, right_lbls = [], [], [], []
    for c in fam:
        items = by_cls.get(c, [])[: args.mine_per_class]
        if not items:
            continue
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        cache = [b[0]["pixel_values"] for b in dl]
        pred = run(cache)
        for i, it in enumerate(items):
            (wrong_items if pred[i] != c else right_items).append(it)
            (wrong_lbls if pred[i] != c else right_lbls).append(c)
    print(f"\nmined {len(wrong_items)} errors and {len(right_items)} correct videos "
          f"on {len(fam)} family classes")

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    wcache, wl = cache_of(wrong_items), np.array(wrong_lbls)
    rcache, rl = cache_of(right_items), np.array(right_lbls)
    # other-class collateral sample
    other = [it for it in val if label2idx.get(it.get("template", ""), -1) not in set(fam)][:200]
    olbl = np.array([label2idx[it["template"]] for it in other])
    ocache = cache_of(other)
    o_base = float((run(ocache) == olbl).mean())

    print("\n=== repair rates (previously-wrong family videos) ===")
    res = {"family_classes": [lab(c) for c in fam], "n_wrong": len(wrong_items),
           "n_right": len(right_items), "conditions": {}}
    best = (None, -1)
    for name, idx in sets:
        for a in args.alphas:
            pred = run(wcache, patch_sets=(idx, a))
            rate = float((pred == wl).mean())
            res["conditions"][f"{name}@a{a}"] = round(rate, 3)
            print(f"  {name:<12} alpha={a:<4} repair rate = {rate:.3f}")
            if name in ("NFP-85", "flippers-12") and rate > best[1]:
                best = ((name, idx, a), rate)
    # collateral for the best NFP condition
    (bname, bidx, balpha), brate = best
    keep = float((run(rcache, patch_sets=(bidx, balpha)) == rl).mean())
    o_amp = float((run(ocache, patch_sets=(bidx, balpha)) == olbl).mean())
    print(f"\nbest temporal condition: {bname} alpha={balpha} (repair {brate:.3f})")
    print(f"  collateral: previously-correct family videos stay correct: {keep:.3f}")
    print(f"  collateral: other-class sample accuracy {o_base:.3f} -> {o_amp:.3f}")
    res["best"] = {"cond": f"{bname}@a{balpha}", "repair": round(brate, 3),
                   "family_keep": round(keep, 3),
                   "other_acc_base": round(o_base, 3), "other_acc_amp": round(o_amp, 3)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
