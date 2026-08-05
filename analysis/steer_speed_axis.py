"""
Experiment A — bidirectional, within-class speed-axis steering (the robust 4-cell test).

Motivation: a feature's single top-activation class is an ambiguous steering target -- it may be
a fast class (rolling) or a slow one (gently moving). Steering "toward that class" conflates
"the feature adds speed" with "the feature is an attractor for one label". To separate them we
hold input CONTENT fixed and steer the feature UP and DOWN, starting from both a fast and a slow
class, and read out a content-independent speed scalar.

Readout: with a data-defined per-class speed score (optical flow; class_speed.json),
  E[speed | prediction] = sum_c P(c) * speed_z(c).
A genuine speed feature raises E[speed] when clamped UP and lowers it when clamped DOWN,
consistently from BOTH a fast starting class (C_hi) and a slow one (C_lo) -- 4 cells:

         input C_hi (fast)      input C_lo (slow)
  UP      E[speed] should rise   E[speed] should rise
  DOWN    E[speed] should drop   E[speed] should drop

C_hi / C_lo for a feature = the highest- / lowest-speed class among its concept classes
(GT classes of its top-activating clips). We sweep the clamp s and fit the slope
dE[speed]/ds in each input set; a feature "passes" if the slope is positive in both sets and
beats a random-feature null, and if the 4 directional cells have the expected signs.

Usage (from sae-for-vlm/):
  python analysis/steer_speed_axis.py --n_videos 384 --per_class 12 --s_grid -50 0 50 100 150
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


class ItemFrames(SSv2Frames):
    """SSv2Frames over an explicit pre-selected list of val items."""
    def __init__(self, videos_dir, items):
        self.items = items
        self.dir = Path(videos_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--class_speed", default="local_runs/steering/class_speed.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--s_grid", nargs="*", type=float, default=[-50, 0, 50, 100, 150])
    ap.add_argument("--top_clips", default=25, type=int)
    ap.add_argument("--per_class", default=12, type=int, help="input videos per C_hi/C_lo class")
    ap.add_argument("--n_random", default=30, type=int)
    ap.add_argument("--n_videos", default=384, type=int, help="pool for defining concepts")
    ap.add_argument("--min_speed_gap", default=0.5, type=float,
                    help="require z-speed(C_hi)-z-speed(C_lo) >= this, else skip feature")
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_speed_axis.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    label2idx = {v: int(k) for k, v in id2label.items()}
    n_cls = len(id2label)

    cs = json.load(open(args.class_speed))["by_idx"]
    speed_z = np.zeros(n_cls)
    scored = np.zeros(n_cls, dtype=bool)
    for c, v in cs.items():
        speed_z[int(c)] = v["speed_z"]; scored[int(c)] = True
    print(f"class speed loaded: {scored.sum()}/{n_cls} classes scored")

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p = nfp["p_val"].numpy(); t = nfp["t_stat"].numpy()
    bonf = 0.05 / p.shape[0]
    TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
    sig = [int(i) for i in np.where((p < bonf).any(1))[0]]
    dom = {i: TAU[int(np.argmax(np.abs(t[i])))] for i in sig}
    rng = np.random.RandomState(args.seed)
    rand_feats = rng.choice([k for k in range(sae.dict_size) if k not in set(sig)],
                            size=args.n_random, replace=False).tolist()

    # --- 1. concept classes from a base pool ---
    pool = SSv2Frames(args.ssv2_videos, args.ssv2_val_json, args.n_videos, args.seed)
    pdl = DataLoader(pool, batch_size=args.batch_size, shuffle=False, num_workers=0,
                     collate_fn=ssv2_collate(proc))
    pcache, pgt = [], []
    for b in pdl:
        pcache.append(b[0]["pixel_values"]); pgt += [label2idx.get(tm, -1) for tm in b[2]]
    pgt = np.array(pgt)
    steer.enabled = False; steer.record = True; steer.captured = []
    for pv in pcache:
        with torch.no_grad():
            clf(pixel_values=pv.to(device))
    A = torch.cat(steer.captured, 0).numpy(); steer.record = False
    print(f"concept activations: A={A.shape}")

    def hi_lo(k):
        """C_hi, C_lo = highest / lowest speed scored concept class of feature k."""
        order = np.argsort(-A[:, k])
        cc = [pgt[j] for j in order[: args.top_clips] if pgt[j] >= 0 and scored[pgt[j]]]
        if len(cc) < 2:
            return None
        uniq = list(dict.fromkeys(cc))  # preserve, dedup
        if len(uniq) < 2:
            return None
        hi = max(uniq, key=lambda c: speed_z[c])
        lo = min(uniq, key=lambda c: speed_z[c])
        if speed_z[hi] - speed_z[lo] < args.min_speed_gap:
            return None
        return int(hi), int(lo)

    # --- 2. cache input videos for every needed class ---
    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_cls = {}
    for it in val:
        c = label2idx.get(it.get("template", ""), -1)
        if c >= 0:
            by_cls.setdefault(c, []).append(it)

    feat_hl = {}
    for k in sig + rand_feats:
        hl = hi_lo(k)
        if hl and len(by_cls.get(hl[0], [])) >= 4 and len(by_cls.get(hl[1], [])) >= 4:
            feat_hl[k] = hl
    needed = sorted({c for hl in feat_hl.values() for c in hl})
    print(f"{len(feat_hl)} features have a valid C_hi/C_lo pair | {len(needed)} distinct classes to cache")

    class_cache = {}
    for c in needed:
        items = by_cls[c][: args.per_class]
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        class_cache[c] = [b[0]["pixel_values"] for b in dl]

    def e_speed(cache):
        """mean over clips of E[speed|pred]."""
        vals = []
        for pv in cache:
            with torch.no_grad():
                P = torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu().numpy()
            vals.append(P @ speed_z)
        return float(np.concatenate(vals).mean())

    def curve(cache, k):
        steer.k = k
        ys = []
        for s in args.s_grid:
            steer.enabled = True; steer.s = s
            ys.append(e_speed(cache));
        steer.enabled = False
        return np.array(ys)

    sg = np.array(args.s_grid)
    def slope(ys):
        return float(np.polyfit(sg, ys, 1)[0])
    s0 = list(args.s_grid).index(0) if 0 in args.s_grid else 0
    smax = len(args.s_grid) - 1

    def eval_feat(k):
        hi, lo = feat_hl[k]
        y_hi = curve(class_cache[hi], k)
        y_lo = curve(class_cache[lo], k)
        # 4 cells: up = S(max)-S(0) ; down = S(min)-S(0)
        up_hi, dn_hi = y_hi[smax] - y_hi[s0], y_hi[0] - y_hi[s0]
        up_lo, dn_lo = y_lo[smax] - y_lo[s0], y_lo[0] - y_lo[s0]
        cells = {"hi_up": bool(up_hi > 0), "hi_dn": bool(dn_hi < 0),
                 "lo_up": bool(up_lo > 0), "lo_dn": bool(dn_lo < 0)}
        return {"feature": f"feat{k:05d}", "idx": k, "dom_tau": dom.get(k, "rand"),
                "C_hi": lab(hi), "C_lo": lab(lo),
                "speed_hi": round(float(speed_z[hi]), 2), "speed_lo": round(float(speed_z[lo]), 2),
                "slope_hi": round(slope(y_hi), 4), "slope_lo": round(slope(y_lo), 4),
                "y_hi": [round(float(x), 3) for x in y_hi],
                "y_lo": [round(float(x), 3) for x in y_lo],
                "cells": cells, "n_cells_pass": int(sum(cells.values())),
                "both_slopes_pos": bool(slope(y_hi) > 0 and slope(y_lo) > 0)}

    print("\n=== significant features (4-cell speed-axis steering) ===")
    sig_rows = []
    for k in [f for f in sig if f in feat_hl]:
        r = eval_feat(k); sig_rows.append(r)
        c = r["cells"]
        flag = "".join(x for x, ok in [("H+", c["hi_up"]), ("H-", c["hi_dn"]),
                                       ("L+", c["lo_up"]), ("L-", c["lo_dn"])] if ok)
        print(f"  {r['feature']} [{r['dom_tau']:9s}] slope hi={r['slope_hi']:+.3f} lo={r['slope_lo']:+.3f} "
              f"cells={r['n_cells_pass']}/4 [{flag:8s}]  {r['C_hi'][:22]} / {r['C_lo'][:22]}")

    print("\n=== random-feature null ===")
    rnd_rows = []
    for k in [f for f in rand_feats if f in feat_hl]:
        r = eval_feat(k); rnd_rows.append(r)
    rnd_slopes = np.array([(r["slope_hi"] + r["slope_lo"]) / 2 for r in rnd_rows]) if rnd_rows else np.array([0.0])
    rmean, rsd = float(rnd_slopes.mean()), float(rnd_slopes.std())
    print(f"random mean-slope={rmean:+.4f} sd={rsd:.4f} (n={len(rnd_rows)})")

    for r in sig_rows:
        r["mean_slope"] = round((r["slope_hi"] + r["slope_lo"]) / 2, 4)
        r["slope_z"] = round((r["mean_slope"] - rmean) / (rsd + 1e-9), 2)

    sig_rows.sort(key=lambda r: -r["mean_slope"])
    n_all4 = sum(1 for r in sig_rows if r["n_cells_pass"] == 4)
    n_ge3 = sum(1 for r in sig_rows if r["n_cells_pass"] >= 3)
    n_bothpos = sum(1 for r in sig_rows if r["both_slopes_pos"])
    n_z2 = sum(1 for r in sig_rows if r["slope_z"] >= 2 and r["both_slopes_pos"])

    out = {"s_grid": args.s_grid, "n_eval": len(sig_rows),
           "rand_slope_mean": rmean, "rand_slope_sd": rsd,
           "n_all_4_cells": n_all4, "n_ge_3_cells": n_ge3,
           "n_both_slopes_pos": n_bothpos, "n_slope_z2_and_pos": n_z2,
           "features": sig_rows, "random": rnd_rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n--- COUNTS (n_eval={len(sig_rows)} of 85 with valid C_hi/C_lo) ---")
    print(f"  pass ALL 4 directional cells      : {n_all4}/{len(sig_rows)}")
    print(f"  pass >=3 of 4 cells               : {n_ge3}/{len(sig_rows)}")
    print(f"  positive slope in BOTH input sets : {n_bothpos}/{len(sig_rows)}")
    print(f"  both-pos AND slope_z>=2 vs random : {n_z2}/{len(sig_rows)}  <-- robust speed steerers")
    print(f"\n  top 15 by mean slope:")
    for r in sig_rows[:15]:
        print(f"    {r['feature']} [{r['dom_tau']:9s}] mslope={r['mean_slope']:+.3f} z={r['slope_z']:+.1f} "
              f"cells={r['n_cells_pass']}/4  {r['C_hi'][:20]} / {r['C_lo'][:20]}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
