"""
Experiment A — generalization test for the PROVEN motion-knob features.

Test C (steer_speed_axis.py) proved ~9 features are direction-consistent motion knobs on THEIR
OWN two anchor classes (C_hi/C_lo drawn from each feature's concept set). This asks the next
question: does the steering effect PERSIST on input classes the feature was NOT tuned on? If a
feature steers up motion starting from "rolling", does it also do so starting from "pulling",
"holding", "throwing", ...?

Design: take the proven features. Build a DIVERSE PANEL of input classes spanning the whole motion
spectrum (slow -> fast), excluding "Turning the camera ..." classes (whole-frame camera motion is
a known confound of the optical-flow axis). For each proven feature x each panel class: hold that
class's real videos as fixed input, sweep the clamp s, and measure the motion readout
E[speed|pred] = sum_c P(c) * speed_z(c). A GENUINE general motion knob raises E[speed] when
steered up FROM EVERY starting class; an attractor only raises it from slow starts and lowers it
from fast starts. So the headline metric is:

  up_frac(feature) = fraction of panel classes where E[speed]@s_max > E[speed]@s0   (in [0,1])

up_frac ~ 1.0  => the knob generalizes across content;  ~0.5 => attractor / content-dependent.

Usage (from sae-for-vlm/):
  python analysis/steer_generalize.py --panel_size 12 --per_class 12 --s_grid 0 50 100 150
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
    def __init__(self, videos_dir, items):
        self.items = items
        self.dir = Path(videos_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--speed_axis", default="local_runs/steering/expA_speed_axis.json",
                    help="Test C results; proven features read from here")
    ap.add_argument("--class_speed", default="local_runs/steering/class_speed.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--features", nargs="*", type=int, default=None,
                    help="override: explicit feature indices; default = robust 9 from Test C")
    ap.add_argument("--s_grid", nargs="*", type=float, default=[0, 50, 100, 150])
    ap.add_argument("--panel_size", default=12, type=int)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--n_random", default=15, type=int, help="random features for a null up_frac")
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_generalize.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    label2idx = {v: int(k) for k, v in id2label.items()}
    n_cls = len(id2label)

    cs = json.load(open(args.class_speed))["by_idx"]
    speed_z = np.zeros(n_cls); scored = np.zeros(n_cls, dtype=bool)
    for c, v in cs.items():
        speed_z[int(c)] = v["speed_z"]; scored[int(c)] = True

    # proven features: both slopes positive AND slope_z >= 2 (unless overridden)
    axis = json.load(open(args.speed_axis))
    if args.features:
        proven = [{"idx": k, "dom_tau": "override"} for k in args.features]
    else:
        proven = [{"idx": r["idx"], "dom_tau": r["dom_tau"]}
                  for r in axis["features"] if r["both_slopes_pos"] and r["slope_z"] >= 2]
    proven_idx = [p["idx"] for p in proven]
    print("proven features (from Test C): " + ", ".join(f"feat{i:05d}" for i in proven_idx))

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    rng = np.random.RandomState(args.seed)
    rand_feats = rng.choice([k for k in range(sae.dict_size) if k not in set(proven_idx)],
                            size=args.n_random, replace=False).tolist()

    # ---- build diverse panel of input classes (exclude camera-pan classes) ----
    val = json.load(open(args.ssv2_val_json))
    by_cls = {}
    for it in val:
        c = label2idx.get(it.get("template", ""), -1)
        if c >= 0:
            by_cls.setdefault(c, []).append(it)
    cand = [c for c in range(n_cls)
            if scored[c] and len(by_cls.get(c, [])) >= args.per_class
            and "Turning the camera" not in lab(c)]
    cand.sort(key=lambda c: speed_z[c])
    if len(cand) <= args.panel_size:
        panel = cand
    else:
        pick = np.linspace(0, len(cand) - 1, args.panel_size).round().astype(int)
        panel = [cand[i] for i in sorted(set(pick.tolist()))]
    print(f"\npanel = {len(panel)} input classes spanning speed_z "
          f"[{speed_z[panel[0]]:+.2f} .. {speed_z[panel[-1]]:+.2f}]:")
    for c in panel:
        print(f"   z{speed_z[c]:+.2f}  {lab(c)}")

    class_cache = {}
    for c in panel:
        items = by_cls[c][: args.per_class]
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        class_cache[c] = [b[0]["pixel_values"] for b in dl]

    sg = np.array(args.s_grid)
    s0 = list(args.s_grid).index(0) if 0 in args.s_grid else 0

    def e_speed(cache):
        vals = []
        for pv in cache:
            with torch.no_grad():
                P = torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu().numpy()
            vals.append(P @ speed_z)
        return float(np.concatenate(vals).mean())

    def feature_over_panel(k):
        per_class = []
        for c in panel:
            steer.k = k
            ys = []
            for s in args.s_grid:
                steer.enabled = (s != 0); steer.s = s
                ys.append(e_speed(class_cache[c]))
            steer.enabled = False
            ys = np.array(ys)
            slope = float(np.polyfit(sg, ys, 1)[0])
            up = bool(ys[-1] > ys[s0])
            per_class.append({"cls": lab(c), "speed_z": round(float(speed_z[c]), 2),
                              "slope": round(slope, 4), "up": up,
                              "e0": round(float(ys[s0]), 3), "emax": round(float(ys[-1]), 3)})
        up_frac = float(np.mean([r["up"] for r in per_class]))
        mean_slope = float(np.mean([r["slope"] for r in per_class]))
        pos_slope_frac = float(np.mean([r["slope"] > 0 for r in per_class]))
        return {"up_frac": round(up_frac, 3), "mean_slope": round(mean_slope, 4),
                "pos_slope_frac": round(pos_slope_frac, 3), "per_class": per_class}

    print("\n=== PROVEN features across the diverse panel ===")
    rows = []
    for p in proven:
        r = feature_over_panel(p["idx"]); r.update(p)
        r["feature"] = f"feat{p['idx']:05d}"
        rows.append(r)
        print(f"  {r['feature']} [{p['dom_tau']:9s}] up_frac={r['up_frac']:.2f} "
              f"pos_slope_frac={r['pos_slope_frac']:.2f} mean_slope={r['mean_slope']:+.4f}")

    print("\n=== random-feature null (same panel) ===")
    null_up = []
    for k in rand_feats:
        r = feature_over_panel(k); null_up.append(r["up_frac"])
    null_up = np.array(null_up)
    print(f"random up_frac: mean={null_up.mean():.3f} sd={null_up.std():.3f} (n={len(null_up)})")

    rows.sort(key=lambda r: -r["up_frac"])
    n_generalize = sum(1 for r in rows if r["up_frac"] >= 0.75)
    out = {"s_grid": args.s_grid, "panel": [lab(c) for c in panel],
           "panel_speed_z": [round(float(speed_z[c]), 2) for c in panel],
           "n_proven": len(rows), "n_generalize_up075": n_generalize,
           "rand_up_frac_mean": float(null_up.mean()), "rand_up_frac_sd": float(null_up.std()),
           "features": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n--- SUMMARY ---")
    print(f"  proven features tested            : {len(rows)}")
    print(f"  generalize (up_frac >= 0.75)      : {n_generalize}/{len(rows)}")
    print(f"  random-feature up_frac            : {null_up.mean():.2f} +/- {null_up.std():.2f}")
    print(f"\n  per-feature (sorted by up_frac):")
    for r in rows:
        print(f"    {r['feature']} [{r['dom_tau']:9s}] up_frac={r['up_frac']:.2f} "
              f"mean_slope={r['mean_slope']:+.4f}")
    # show the best feature's full per-class breakdown (answers "rolling -> pulling?")
    if rows:
        best = rows[0]
        print(f"\n  {best['feature']} per-class detail (does it persist across content?):")
        for pc in sorted(best["per_class"], key=lambda x: x["speed_z"]):
            mark = "UP  " if pc["up"] else "down"
            print(f"    [{mark}] slope={pc['slope']:+.4f}  z{pc['speed_z']:+.2f}  "
                  f"{pc['e0']:.2f}->{pc['emax']:.2f}  {pc['cls'][:40]}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
