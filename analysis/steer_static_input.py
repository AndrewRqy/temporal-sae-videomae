"""
Experiment A, faithful LLaVA analog — steer on a MOTIONLESS input.

The cleanest test of "can a temporal SAE feature inject its concept": feed the classifier a
frozen video (one real frame repeated 16x, so there is zero motion), where it defaults to
static classes ("Holding", "Showing ... to the camera"). Then clamp a feature on all tokens
and see whether probability mass moves toward MOTION classes. On a motionless input any motion
the classifier reports was injected by the steering — there is no real motion to confound it.
This mirrors the LLaVA "white image + steer a neuron -> concept appears in the output" demo.

Readouts per feature:
  d_motion   = sum_{c in MOTION} dP_c - sum_{c in NOMOTION} dP_c   (did it inject dynamism?)
  TV         = 0.5 * sum_c |dP_c|                                  (how much the output moved)
  top rising / falling classes
vs a random-feature null. Reports which features most inject motion, and toward what.

Usage (from sae-for-vlm/):
  python analysis/steer_static_input.py --n_videos 256 --s_abs 100 --frame first
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from analysis.steer_ssv2_logits import SteerLayer, ssv2_collate

# Classes implying object dynamics vs near-static "presentation" classes (matched on id2label).
MOTION_KW = ["throwing", "pushing", "pulling", "moving", "rolling", "spinning", "falling",
             "dropping", "lifting", "tipping", "putting", "taking", "twisting", "bending",
             "tearing", "poking", "hitting", "squeezing", "pouring", "spilling", "stuffing",
             "scooping", "knocking", "colliding", "sprinkling", "piling", "stacking"]
NOMOTION_KW = ["holding", "showing", "pretending", "without moving", "approaching",
               "so lightly", "almost doesn't", "without letting"]


class StaticFrames(Dataset):
    """One real frame of each SSv2 clip, repeated 16x -> a motionless clip."""
    def __init__(self, videos_dir, val_json, n, seed=0, frame="first"):
        val = json.load(open(val_json))
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(val), size=min(n, len(val)), replace=False)
        self.items = [val[i] for i in sel]
        self.dir = Path(videos_dir)
        self.frame = frame

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import av
        it = self.items[i]
        try:
            cont = av.open(str(self.dir / f"{it['id']}.webm"))
            frames = [f.to_ndarray(format="rgb24") for f in cont.decode(video=0)]
            cont.close()
            fr = frames[0] if self.frame == "first" else frames[len(frames) // 2]
            img = Image.fromarray(fr)
        except Exception:
            img = Image.new("RGB", (224, 224))
        return [img] * 16, it["id"], it.get("template", "")


def matched(id2label, kws):
    out = []
    for i, l in id2label.items():
        L = l.lower()
        if any(k in L for k in kws):
            out.append(int(i))
    return out


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
    ap.add_argument("--frame", default="first", choices=["first", "middle"])
    ap.add_argument("--n_random", default=30, type=int)
    ap.add_argument("--n_videos", default=256, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_static_input.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    motion = np.array(matched(id2label, MOTION_KW))
    nomotion = np.array(matched(id2label, NOMOTION_KW))
    print(f"MOTION classes={len(motion)} NOMOTION classes={len(nomotion)} | clamp s={args.s_abs} | frame={args.frame}")

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

    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    ds = StaticFrames(args.ssv2_videos, args.ssv2_val_json, args.n_videos, args.seed, args.frame)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=ssv2_collate(proc))
    cached = [b[0]["pixel_values"] for b in dl]
    print(f"cached {sum(b.shape[0] for b in cached)} motionless clips")

    def all_probs():
        outs = []
        for pv in cached:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        return torch.cat(outs, 0).numpy()

    steer.enabled = False
    base = all_probs()
    bm = base.mean(0)
    print("baseline (static) top-5 classes:", [(lab(i), round(float(bm[i]), 3)) for i in np.argsort(-bm)[:5]])

    def measure(k):
        steer.enabled = True; steer.k = k; steer.s = args.s_abs
        P = all_probs(); steer.enabled = False
        dP = (P - base).mean(0)
        d_motion = float(dP[motion].sum() - dP[nomotion].sum())
        tv = float(0.5 * np.abs(dP).sum())
        up = np.argsort(-dP)[:3]; dn = np.argsort(dP)[:3]
        return {"d_motion": d_motion, "tv": tv,
                "up": [(lab(i), round(float(dP[i]), 4)) for i in up],
                "down": [(lab(i), round(float(dP[i]), 4)) for i in dn]}

    print("\n=== random-feature null ===")
    null = [dict(measure(k), feature=f"rand{k}") for k in rand_feats]
    for m in null:
        print(f"  {m['feature']:<10} d_motion={m['d_motion']:+.4f} tv={m['tv']:.4f}")
    nd = np.array([m["d_motion"] for m in null])
    nm, ns = nd.mean(), nd.std() + 1e-9
    print(f"null d_motion: mean={nm:+.4f} std={ns:.4f}")

    print("\n=== significant temporal features ===")
    rows = []
    for k in sig:
        m = measure(k)
        m.update({"feature": f"feat{k:05d}", "idx": k, "dom_tau": dom[k],
                  "z_motion": round((m["d_motion"] - nm) / ns, 2)})
        rows.append(m)
        print(f"  feat{k:05d} [{dom[k]:9s}] d_motion={m['d_motion']:+.4f} z={m['z_motion']:+5.1f} "
              f"tv={m['tv']:.3f}  up:{m['up'][0]}")

    rows.sort(key=lambda r: -r["z_motion"])
    out = {"s_abs": args.s_abs, "frame": args.frame, "n_clips": int(base.shape[0]),
           "baseline_top": [(lab(i), float(bm[i])) for i in np.argsort(-bm)[:8]],
           "null_motion_mean": float(nm), "null_motion_std": float(ns),
           "motion_classes": [lab(i) for i in motion], "features": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    win = [r for r in rows if r["z_motion"] >= 3]
    print(f"\n=== {len(win)}/{len(rows)} features inject motion (z>=3) ===")
    for r in win[:25]:
        print(f"  {r['feature']} [{r['dom_tau']:9s}] z={r['z_motion']:+5.1f}  up:{r['up'][0]}  down:{r['down'][0]}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
