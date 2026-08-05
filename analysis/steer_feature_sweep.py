"""
Experiment A, broadened — steer MANY SAE features and see how many steer successfully.

One working feature could be luck. This sweeps every NFP-significant temporal feature (85),
clamps each (one at a time) to a fixed value s on all 1568 layer-11 tokens, and measures how
the SSv2 classifier's output distribution moves, vs a null of random (non-significant)
features clamped to the same s (so the comparison is magnitude-matched by construction —
same absolute clamp for everyone, the CLIP-ViT / sae-for-vlm recipe).

Per feature we record:
  S_f         = mean over clips of ||P_steered - P_base||_2^2   (general steerability; CLIP-ViT metric)
  force_shift = sum_{c in FAST} dP_c - sum_{c in SLOW} dP_c      (on the fast/forceful axis)
  top rising / falling classes                                  (what concept it steers toward)
Then z-score S_f and force_shift against the random-feature null. "Successful" = output moves
far outside the null with a coherent semantic shift.

Usage (from sae-for-vlm/):
  python analysis/steer_feature_sweep.py --n_videos 256 --s_abs 100 --n_random 30
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
from analysis.steer_ssv2_logits import (
    SteerLayer, SSv2Frames, ssv2_collate, build_fast_slow,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--candidates_csv", default="local_runs/nfp_feature_gifs/steering_candidates.csv")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--s_abs", default=100.0, type=float, help="fixed clamp value for every feature")
    ap.add_argument("--n_random", default=30, type=int)
    ap.add_argument("--n_videos", default=256, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expA_feature_sweep.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    id2label = clf.config.id2label
    lab = lambda i: id2label.get(str(i), id2label.get(i))
    fast, slow = build_fast_slow(id2label)
    print(f"SSv2 classes 174 | FAST {len(fast)} SLOW {len(slow)} | clamp s={args.s_abs}")

    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer

    # significant temporal features + a label of which tau each is dominant on
    nfp = torch.load(args.nfp_results, map_location="cpu")
    p = nfp["p_val"].numpy(); t = nfp["t_stat"].numpy()
    D = p.shape[0]; bonf = 0.05 / D
    TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
    sig = np.where((p < bonf).any(1))[0]
    dom = {int(i): TAU[int(np.argmax(np.abs(t[i])))] for i in sig}
    print(f"{len(sig)} significant features to steer")
    rng = np.random.RandomState(args.seed)
    rand_feats = rng.choice([k for k in range(sae.dict_size) if k not in set(sig.tolist())],
                            size=args.n_random, replace=False).tolist()

    # decode SSv2-val once, cache pixel_values
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)
    ds = SSv2Frames(args.ssv2_videos, args.ssv2_val_json, args.n_videos, args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=ssv2_collate(proc))
    cached = [b[0]["pixel_values"] for b in dl]
    n_clips = sum(b.shape[0] for b in cached)
    print(f"cached {n_clips} SSv2-val clips")
    fast_i, slow_i = np.array(fast), np.array(slow)

    def all_probs():
        outs = []
        for pv in cached:
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
        return torch.cat(outs, 0).numpy()

    steer.enabled = False
    base = all_probs()
    print(f"baseline ready ({base.shape[0]} clips)")

    def measure(k):
        steer.enabled = True; steer.k = k; steer.s = args.s_abs
        P = all_probs(); steer.enabled = False
        dP = P - base
        S_f = float((dP ** 2).sum(1).mean())                       # per-clip L2^2, averaged
        fshift = float(dP[:, fast_i].sum(1).mean() - dP[:, slow_i].sum(1).mean())
        md = dP.mean(0)
        up = np.argsort(-md)[:3]; dn = np.argsort(md)[:3]
        return {"S_f": S_f, "force_shift": fshift,
                "up": [(lab(i), round(float(md[i]), 4)) for i in up],
                "down": [(lab(i), round(float(md[i]), 4)) for i in dn]}

    print("\n=== random-feature null ===")
    null = []
    for k in rand_feats:
        m = measure(k); null.append(m)
        print(f"  rand{k:<5} S_f={m['S_f']:.4f} fshift={m['force_shift']:+.4f}")
    nullS = np.array([m["S_f"] for m in null])
    nullF = np.array([m["force_shift"] for m in null])
    Smu, Ssd = nullS.mean(), nullS.std() + 1e-9
    Fmu, Fsd = nullF.mean(), nullF.std() + 1e-9
    print(f"null S_f: mean={Smu:.4f} std={Ssd:.4f} | force_shift: mean={Fmu:+.4f} std={Fsd:.4f}")

    print("\n=== significant temporal features ===")
    rows = []
    for k in sig:
        k = int(k); m = measure(k)
        m.update({"feature": f"feat{k:05d}", "idx": k, "dom_tau": dom[k],
                  "z_Sf": round((m["S_f"] - Smu) / Ssd, 2),
                  "z_force": round((m["force_shift"] - Fmu) / Fsd, 2)})
        rows.append(m)
        print(f"  feat{k:05d} [{dom[k]:9s}] S_f={m['S_f']:.4f} (z={m['z_Sf']:+5.1f}) "
              f"fshift={m['force_shift']:+.4f} (z={m['z_force']:+5.1f})  up:{m['up'][0]}")

    rows.sort(key=lambda r: -r["z_Sf"])
    out = {"s_abs": args.s_abs, "n_clips": int(base.shape[0]),
           "null": {"Sf_mean": float(Smu), "Sf_std": float(Ssd),
                    "force_mean": float(Fmu), "force_std": float(Fsd), "features": rand_feats},
           "features": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    steer_thresh = Smu + 3 * Ssd
    winners = [r for r in rows if r["S_f"] > steer_thresh]
    print(f"\n=== {len(winners)}/{len(rows)} features steerable (S_f > null mean+3std) ===")
    for r in winners[:20]:
        print(f"  {r['feature']} [{r['dom_tau']:9s}] z_Sf={r['z_Sf']:+6.1f} z_force={r['z_force']:+6.1f}  "
              f"up:{r['up'][0]}  down:{r['down'][0]}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
