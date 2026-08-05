"""
Experiment B mechanism — decompose the (fully steering-capable) direction axis onto the
SAE dictionary.

steer_direction_diffmeans.py showed a single supervised layer-11 vector v flips every
direction pair at 100%, while single SAE features barely move the horizontal pairs. Why?
Decompose v in the dictionary: cosine of v with every decoder column n_k, and the
fraction of ||v||^2 explained by the best m columns (least squares on top-|cos| columns).
If the mass is spread thin — or concentrated in features NFP never flagged — that
explains the gap and says something real about what the SAE did to the direction axis.

Usage (from sae-for-vlm/):
  python analysis/decompose_direction_axis.py --n_build 24
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
from analysis.steer_direction_flip import PAIRS, ItemFrames

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_build", default=24, type=int)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expB_axis_decomposition.json")
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
    sig_set = set(int(i) for i in np.where((p_all < bonf).any(1))[0])
    dom = {i: TAU[int(np.argmax(np.abs(t_all[i])))] for i in sig_set}

    # decoder columns, unit-normalized
    Wd = sae.decoder.weight.data.cpu().float()            # [768, 6144]
    n_unit = Wd / (Wd.norm(dim=0, keepdim=True) + 1e-12)  # [768, 6144]

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    def mean_raw(items):
        steer.captured_raw = []; steer.record_raw = True
        for pv in cache_of(items):
            with torch.no_grad():
                clf(pixel_values=pv.to(device))
        steer.record_raw = False
        return torch.cat(steer.captured_raw, 0).mean(0)   # [768]

    results = {"pairs": []}
    for pair in PAIRS:
        pos_l, neg_l = pair["pos"], pair["neg"]
        if len(by_tmpl.get(pos_l, [])) < args.n_build or len(by_tmpl.get(neg_l, [])) < args.n_build:
            continue
        print(f"\n=== AXIS  {pos_l}  -  {neg_l} ===")
        v = mean_raw(by_tmpl[pos_l][: args.n_build]) - mean_raw(by_tmpl[neg_l][: args.n_build])
        v_hat = (v / v.norm()).float()                     # [768]

        cos = (v_hat @ n_unit).numpy()                     # [6144]
        order = np.argsort(-np.abs(cos))
        print(f"  ||v||={v.norm():.1f}   top-10 |cos(v, n_k)|:")
        top_rows = []
        for r, k in enumerate(order[:10]):
            k = int(k)
            in_sig = k in sig_set
            tag = dom.get(k, "-")
            print(f"    #{r+1:2d} feat{k:05d}  cos={cos[k]:+.3f}  "
                  f"NFP-sig={'YES ('+tag+')' if in_sig else 'no'}")
            top_rows.append({"feature": f"feat{k:05d}", "idx": k,
                             "cos": round(float(cos[k]), 4),
                             "nfp_sig": bool(in_sig), "dom_tau": tag if in_sig else None})

        # fraction of ||v||^2 explained by best-m columns (least squares on top-|cos| m)
        expl = {}
        for m in [1, 5, 20, 100, 500]:
            cols = torch.tensor(order[:m].copy())
            B = n_unit[:, cols]                            # [768, m]
            coef = torch.linalg.lstsq(B, v_hat.unsqueeze(1)).solution
            r2 = float(1 - (v_hat.unsqueeze(1) - B @ coef).norm() ** 2)
            expl[m] = round(r2, 4)
        print("  R^2 of v on top-m decoder columns: "
              + "  ".join(f"m={m}:{r}" for m, r in expl.items()))

        # how much of v lives in the 85 NFP-flagged features' span?
        sig_cols = torch.tensor(sorted(sig_set))
        Bs = n_unit[:, sig_cols]
        coef = torch.linalg.lstsq(Bs, v_hat.unsqueeze(1)).solution
        r2_sig = float(1 - (v_hat.unsqueeze(1) - Bs @ coef).norm() ** 2)
        print(f"  R^2 of v on the {len(sig_set)} NFP-flagged features' span: {r2_sig:.4f}")

        results["pairs"].append({"pos": pos_l, "neg": neg_l, "v_norm": float(v.norm()),
                                 "top10": top_rows, "r2_top_m": expl,
                                 "r2_nfp_span": round(r2_sig, 4)})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
