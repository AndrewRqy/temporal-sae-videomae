"""
Experiment A — strict single-class steering count.

The question: for each NFP-significant feature, define its concept as the SINGLE majority
ground-truth class among its top-activating clips, steer the feature, and measure whether the
probability of THAT ONE class rises. Count how many of the 85 show a meaningful effect, judged
against a random-feature null (random features steered toward their own single top class).

This is stricter than steer_concept_alignment.py (which uses the full top-clip class
distribution as a soft target). Here the target is one class only.

  dP_top(k) = mean_clips [ P_steered(c_k) - P_base(c_k) ],  c_k = argmax class of feat k's top clips
  null: same quantity for n_random random features -> mean, sd
  meaningful if dP_top(k) > null_mean + 2*null_sd  (also report per-feature z and P-multiple)

Usage (from sae-for-vlm/):
  python analysis/steer_topclass.py --n_videos 384 --s_abs 100 --top_clips 25
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--s_abs", default=100.0, type=float)
    ap.add_argument("--top_clips", default=25, type=int)
    ap.add_argument("--n_random", default=40, type=int)
    ap.add_argument("--n_videos", default=384, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_topclass.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    label2idx = {v: int(k) for k, v in id2label.items()}

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p = nfp["p_val"].numpy(); t = nfp["t_stat"].numpy()
    bonf = 0.05 / p.shape[0]
    TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
    sig = [int(i) for i in np.where((p < bonf).any(1))[0]]
    dom = {i: TAU[int(np.argmax(np.abs(t[i])))] for i in sig}
    rng = np.random.RandomState(args.seed)
    rand_feats = rng.choice([k for k in range(sae.dict_size) if k not in set(sig)],
                            size=args.n_random, replace=False).tolist()
    print(f"{len(sig)} significant features | clamp s={args.s_abs} | top_clips={args.top_clips}")

    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    ds = SSv2Frames(args.ssv2_videos, args.ssv2_val_json, args.n_videos, args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=ssv2_collate(proc))
    cached, gt = [], []
    for b in dl:
        cached.append(b[0]["pixel_values"])
        gt += [label2idx.get(tm, -1) for tm in b[2]]
    gt = np.array(gt)
    print(f"cached {sum(b.shape[0] for b in cached)} clips ({(gt>=0).sum()} with known label)")

    def all_probs():
        outs = []
        for pv in cached:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        return torch.cat(outs, 0).numpy()

    steer.enabled = False; steer.record = True; steer.captured = []
    base = all_probs()
    A = torch.cat(steer.captured, 0).numpy()
    steer.record = False
    bm = base.mean(0)
    print(f"baseline + activations captured: A={A.shape}")

    def top_class(k):
        order = np.argsort(-A[:, k])
        top = [gt[j] for j in order[: args.top_clips] if gt[j] >= 0]
        return int(np.bincount(top).argmax()) if top else -1

    def dP_topclass(k, c):
        steer.enabled = True; steer.k = k; steer.s = args.s_abs
        P = all_probs(); steer.enabled = False
        return float((P[:, c] - base[:, c]).mean())

    # random-feature null first (single top class each)
    print("\n=== random-feature null (steer toward own single top class) ===")
    null = []
    for k in rand_feats:
        c = top_class(k)
        if c < 0:
            continue
        d = dP_topclass(k, c)
        null.append(d)
    null = np.array(null)
    nmean, nsd = float(null.mean()), float(null.std())
    print(f"random dP(top class): mean={nmean:+.4f} sd={nsd:.4f}  (n={len(null)})")

    thr = nmean + 2 * nsd
    print(f"\n=== significant features: dP(single top class) at s={args.s_abs} ===")
    rows = []
    for k in sig:
        c = top_class(k)
        if c < 0:
            continue
        d = dP_topclass(k, c)
        z = (d - nmean) / (nsd + 1e-9)
        mult = (bm[c] + d) / (bm[c] + 1e-9)
        rows.append({"feature": f"feat{k:05d}", "idx": k, "dom_tau": dom[k],
                     "concept": lab(c), "p_base": round(float(bm[c]), 5),
                     "dP_top": round(d, 5), "p_steered": round(float(bm[c] + d), 5),
                     "mult": round(float(mult), 2), "z": round(z, 2)})

    rows.sort(key=lambda r: -r["dP_top"])
    n_pos = sum(1 for r in rows if r["dP_top"] > 0)
    n_thr = sum(1 for r in rows if r["dP_top"] > thr)
    n_z3 = sum(1 for r in rows if r["z"] >= 3)
    n_dbl = sum(1 for r in rows if r["mult"] >= 2 and r["dP_top"] > thr)

    out = {"s_abs": args.s_abs, "top_clips": args.top_clips, "n_clips": int(base.shape[0]),
           "null_mean": nmean, "null_sd": nsd, "thr_2sd": thr,
           "n_sig": len(rows), "n_positive": n_pos, "n_above_2sd": n_thr,
           "n_z_ge_3": n_z3, "n_doubled_and_sig": n_dbl, "features": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n--- COUNTS (n_sig={len(rows)}) ---")
    print(f"  dP > 0                         : {n_pos}/{len(rows)}")
    print(f"  dP > null_mean+2sd ({thr:+.4f}) : {n_thr}/{len(rows)}   <-- meaningful vs random")
    print(f"  z >= 3                          : {n_z3}/{len(rows)}")
    print(f"  >=2x base AND > 2sd            : {n_dbl}/{len(rows)}   <-- strong + meaningful")
    print(f"\n  top 20 by dP(top class):")
    for r in rows[:20]:
        print(f"    {r['feature']} [{r['dom_tau']:9s}] dP={r['dP_top']:+.4f} z={r['z']:+5.1f} "
              f"{r['p_base']:.3f}->{r['p_steered']:.3f} ({r['mult']:.1f}x)  {r['concept'][:38]}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
