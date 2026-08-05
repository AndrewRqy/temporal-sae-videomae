"""
Experiment A, definitive readout — LLaVA-faithful concept-alignment steering test.

No hand-built class axis. For each NFP-significant feature we let the DATA define its concept:
the SSv2 classes of the clips that most activate it. Then we steer the feature and ask whether
the classifier's probability mass moves toward THOSE classes, vs a null of random class sets.

Procedure (all on real SSv2-val clips):
  1. Baseline pass records, per clip, the mean-pooled layer-11 SAE activation of every feature.
  2. For feature k: concept_weight_k = normalized histogram of the ground-truth classes of its
     top-Kc activating clips (a distribution over the 174 SSv2 classes).
  3. Steer k (clamp to s on all tokens); dP = mean over clips of (P_steered - P_base).
     alignment_k = sum_c dP_c * concept_weight_k[c]   (did steering raise k's own concept classes?)
  4. Null: align dP_k against many random class-weight vectors (label permutations) -> z-score.
A feature "steers to its own concept" if alignment z is large and positive. Count how many do —
that, not one lucky feature, is the evidence the trick is real.

Usage (from sae-for-vlm/):
  python analysis/steer_concept_alignment.py --n_videos 384 --s_abs 100 --top_clips 25
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
    ap.add_argument("--top_clips", default=25, type=int, help="clips defining each feature's concept")
    ap.add_argument("--n_random", default=30, type=int, help="random features (steering null)")
    ap.add_argument("--n_perm", default=500, type=int, help="label permutations (concept-alignment null)")
    ap.add_argument("--n_videos", default=384, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_concept_alignment.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    label2idx = {v: int(k) for k, v in id2label.items()}
    n_cls = len(id2label)

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

    # baseline + capture per-clip feature activations
    steer.enabled = False; steer.record = True; steer.captured = []
    base = all_probs()
    A = torch.cat(steer.captured, 0).numpy()       # [N, dict] mean-pooled activations
    steer.record = False
    print(f"baseline + activations captured: A={A.shape}")

    def concept_weight(k):
        order = np.argsort(-A[:, k])
        top = [j for j in order[: args.top_clips] if gt[j] >= 0]
        w = np.zeros(n_cls)
        for j in top:
            w[gt[j]] += 1.0
        s = w.sum()
        return (w / s) if s > 0 else None

    def steer_dP(k):
        steer.enabled = True; steer.k = k; steer.s = args.s_abs
        P = all_probs(); steer.enabled = False
        return (P - base).mean(0)                  # [n_cls]

    perm_rng = np.random.RandomState(123)

    def alignment_z(dP, w):
        a = float(dP @ w)
        # null: permute class labels of the weight vector
        idx = np.arange(n_cls)
        null = np.array([float(dP @ w[perm_rng.permutation(idx)]) for _ in range(args.n_perm)])
        z = (a - null.mean()) / (null.std() + 1e-9)
        return a, float(z)

    print("\n=== concept-alignment per significant feature ===")
    rows = []
    for k in sig:
        w = concept_weight(k)
        if w is None:
            continue
        dP = steer_dP(k)
        a, z = alignment_z(dP, w)
        top_concept = [lab(i) for i in np.argsort(-w)[:3] if w[np.argsort(-w)][0] > 0][:3]
        rows.append({"feature": f"feat{k:05d}", "idx": k, "dom_tau": dom[k],
                     "alignment": round(a, 5), "z": round(z, 2),
                     "concept_top": top_concept})
        print(f"  feat{k:05d} [{dom[k]:9s}] align={a:+.4f} z={z:+5.2f}  concept:{top_concept[0]}")

    # random-feature null: same alignment test but concept from the random feature's own top clips
    print("\n=== random-feature null (same test) ===")
    null_z = []
    for k in rand_feats:
        w = concept_weight(k)
        if w is None:
            continue
        a, z = alignment_z(steer_dP(k), w)
        null_z.append(z)
        print(f"  rand{k:<5} z={z:+5.2f}")
    null_z = np.array(null_z)
    print(f"\nrandom-feature alignment-z: mean={null_z.mean():+.2f} std={null_z.std():.2f}")

    rows.sort(key=lambda r: -r["z"])
    thr = 3.0
    winners = [r for r in rows if r["z"] >= thr]
    out = {"s_abs": args.s_abs, "top_clips": args.top_clips, "n_clips": int(base.shape[0]),
           "null_z_mean": float(null_z.mean()), "null_z_std": float(null_z.std()),
           "n_winners": len(winners), "features": rows, "random_null_z": null_z.tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n=== {len(winners)}/{len(rows)} features steer toward their OWN concept (z>={thr}) ===")
    for r in winners[:25]:
        print(f"  {r['feature']} [{r['dom_tau']:9s}] z={r['z']:+5.2f}  concept:{r['concept_top'][0]}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
