"""
Experiment B2 necessity test, sharpened — ablation on HARD (near-boundary) pair videos.

The flat-null ablation result (expB2_controls) probed pair accuracy far from the decision
boundary: baseline margins were |LO| ~ 4.5-5.4, so a <=1-LO margin erosion could not flip any
decision. Here we scan a large pool of videos per pair, select the ones whose baseline pair
log-odds margin is SMALLEST (nearest the boundary, both barely-right and barely-wrong), and
re-run the ablation battery on exactly those. If NFP temporal features carry any necessary
part of the pair signal, decision flips must show up here first.

Conditions (features clamped to 0, decode + re-add error):
  baseline / all-85-temporal / heroes-12 / random-static x85, x12 (2 seeds each) /
  axis-30 = union of the step-14 axis-decomposition top-10 features (the ball-OOD direction
  carriers) — the redundancy account predicts THIS set should bite on its matched pairs.

Metrics per condition, on the hard subset: pair accuracy, mean signed margin LO_own,
correct->wrong and wrong->correct flip counts, and per-video paired margin change
(Wilcoxon temporal-85 vs random-85).

Usage (from sae-for-vlm/):
  python analysis/ablation_hard_videos.py --scan_per_class 48 --keep 12
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from analysis.steer_ssv2_logits import SteerLayer, SSv2Frames, ssv2_collate
from analysis.steer_pair_screen import PAIRS, ItemFrames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--axis_json", default="local_runs/steering/expB_axis_decomposition.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--scan_per_class", default=48, type=int)
    ap.add_argument("--keep", default=12, type=int, help="hardest videos kept per side")
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expB2_ablation_hard.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
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
    heroes = [r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1]
    axis = sorted({f["idx"] for pr in json.load(open(args.axis_json))["pairs"]
                   for f in pr["top10"]})
    print(f"85 temporal | {len(heroes)} heroes | {len(axis)} axis features | "
          f"static pool {len(static_pool)}")

    rng = np.random.RandomState(args.seed + 7)
    sets = [("baseline", None),
            ("temporal x85", sig),
            (f"heroes x{len(heroes)}", heroes),
            (f"axis x{len(axis)}", axis)]
    for si in range(2):
        sets.append((f"rnd-static x85 (s{si})",
                     rng.choice(static_pool, size=len(sig), replace=False).tolist()))
        sets.append((f"rnd-static x{len(heroes)} (s{si})",
                     rng.choice(static_pool, size=len(heroes), replace=False).tolist()))
        sets.append((f"rnd-static x{len(axis)} (s{si})",
                     rng.choice(static_pool, size=len(axis), replace=False).tolist()))

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    def probs(cache):
        outs = []
        for pv in cache:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        return torch.cat(outs, 0).numpy()

    # --- scan: find hardest videos per pair side ---
    print("\nScanning for near-boundary videos...")
    hard = []   # per pair: dict with caches of hard videos + baseline margins
    for key, axis_tag, pos_l, neg_l in PAIRS:
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        if cp < 0 or cn < 0:
            continue
        entry = {"key": key, "cp": cp, "cn": cn, "sides": {}}
        ok = True
        for side, cls_l, own, other in [("pos", pos_l, cp, cn), ("neg", neg_l, cn, cp)]:
            pool = by_tmpl.get(cls_l, [])[: args.scan_per_class]
            if len(pool) < args.keep:
                ok = False; break
            steer.enabled = False
            P = probs(cache_of(pool))
            lo_own = np.log(P[:, own] + 1e-12) - np.log(P[:, other] + 1e-12)
            idx = np.argsort(np.abs(lo_own))[: args.keep]
            items = [pool[i] for i in idx]
            entry["sides"][side] = {"cache": cache_of(items),
                                    "own": own, "other": other,
                                    "base_lo": lo_own[idx]}
        if not ok:
            continue
        b = np.concatenate([entry["sides"][s]["base_lo"] for s in ["pos", "neg"]])
        print(f"  {key:<11} hard-subset baseline: pair-acc={float((b>0).mean()):.2f} "
              f"mean|LO|={float(np.abs(b).mean()):.2f} (vs ~4.5-5.4 for random videos)")
        hard.append(entry)

    # --- ablation battery on the hard subsets ---
    def run_condition(feat_set):
        accs, margins, all_lo = [], [], {}
        for e in hard:
            for side in ["pos", "neg"]:
                sd = e["sides"][side]
                if feat_set is None:
                    P = probs(sd["cache"])
                else:
                    steer.enabled = True; steer.k = feat_set; steer.s = 0.0
                    P = probs(sd["cache"]); steer.enabled = False
                lo = np.log(P[:, sd["own"]] + 1e-12) - np.log(P[:, sd["other"]] + 1e-12)
                all_lo[(e["key"], side)] = lo
                accs.append(float((lo > 0).mean()))
                margins.append(lo)
        m = np.concatenate(margins)
        return float(np.mean(accs)), m, all_lo

    print("\n=== ablation on HARD videos ===")
    results = {"keep": args.keep, "scan_per_class": args.scan_per_class, "conditions": []}
    base_m, base_lo = None, None
    per_cond_m = {}
    for name, fs in sets:
        acc, m, all_lo = run_condition(fs)
        per_cond_m[name] = m
        if name == "baseline":
            base_m, base_lo = m, all_lo
            print(f"  {name:<26} pair-acc={acc:.3f}  mean LO_own={m.mean():+.3f}")
            results["conditions"].append({"set": name, "n": 0, "pair_acc": round(acc, 3),
                                          "mean_lo": round(float(m.mean()), 3)})
            continue
        d = m - base_m
        c2w = int(((base_m > 0) & (m <= 0)).sum())
        w2c = int(((base_m <= 0) & (m > 0)).sum())
        print(f"  {name:<26} pair-acc={acc:.3f}  mean LO_own={m.mean():+.3f}  "
              f"dLO={d.mean():+.3f}  flips correct->wrong={c2w} wrong->correct={w2c}")
        results["conditions"].append({"set": name, "n": len(fs), "pair_acc": round(acc, 3),
                                      "mean_lo": round(float(m.mean()), 3),
                                      "d_lo": round(float(d.mean()), 3),
                                      "c2w": c2w, "w2c": w2c})

    # paired stats: temporal-85 margin damage vs random-85
    t85 = per_cond_m["temporal x85"] - base_m
    for si in range(2):
        r85 = per_cond_m[f"rnd-static x85 (s{si})"] - base_m
        w = stats.wilcoxon(t85, r85, alternative="less")
        print(f"\n  Wilcoxon (temporal-85 dLO < rnd-85 s{si} dLO): p = {w.pvalue:.3e}"
              f"   (temporal {t85.mean():+.3f} vs random {r85.mean():+.3f})")
        results[f"wilcoxon_t85_vs_r85_s{si}"] = float(w.pvalue)
    ax = per_cond_m[[n for n, _ in sets if n.startswith('axis')][0]] - base_m
    for si in range(2):
        rax = per_cond_m[f"rnd-static x{len(axis)} (s{si})"] - base_m
        w = stats.wilcoxon(ax, rax, alternative="less")
        print(f"  Wilcoxon (axis-{len(axis)} dLO < rnd-{len(axis)} s{si} dLO): p = {w.pvalue:.3e}"
              f"   (axis {ax.mean():+.3f} vs random {rax.mean():+.3f})")
        results[f"wilcoxon_axis_vs_rnd_s{si}"] = float(w.pvalue)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
