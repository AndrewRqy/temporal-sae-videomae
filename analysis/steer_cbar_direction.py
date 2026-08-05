"""
Experiment F1 — the maximum-covariance direction c_bar_tau (new theory section 2) as a
steering direction.

Theory: for a linear readout a(V,t) = w^T h(V,t), the direction maximizing the expected
within-video covariance with concept tau is w* = c_bar / ||c_bar|| where

  c_bar_tau = E_V[ (1/T) sum_t (h(V,t) - h_bar(V)) (tau(V,t) - tau_bar(V)) ]   in R^768.

This is closed-form from the cached raw ball-token activations (ball_raw_acts.pt,
3000 x 8 x 768, off-screen steps zeroed exactly as in the NFP pipeline). The theory
note states c_bar is the optimal DETECTION direction and explicitly leaves "is it good
for STEERING?" as an empirical question; the per-unit-delta optimal steering direction
for a class pair (cL, cR) is the head difference (W_head)_cL - (W_head)_cR. This
experiment answers the empirical question.

Protocol:
  1. Compute c_bar for all five taus. Report norms, pairwise cosines, max |cos| to the
     85 NFP features' decoder columns, and cos to the head-diff direction of each
     matched class pair.
  2. Steer by adding delta * unit(c_bar_tau) to all 1568 tokens at layer 11 (add_vec
     mode), delta in {-150, -50, +50, +150}. Matched pairs per tau:
     vel_x -> push_lr, cam_lr; vel_y -> move_ud, cam_ud; speed -> fall_speed;
     accel_mag -> spin_stop; direction -> cam_lr.
     Controls per pair: two random unit directions (seeds 11, 22) and the head-diff
     unit direction (the theory's per-unit-norm task ceiling).
  3. Readout: signed pair log-odds shift Delta = [LO(+150) - LO(-150)]/2 averaged over
     both input sides, and pair-flip rate at the best orientation (as in expB2).
  4. E[speed] dose-response for unit(c_bar_speed) on 24 neutral videos.

Comparison anchors from earlier experiments (same 12-videos/side pairs): best single
NFP feature |LO shift| ~ 1.7 (horizontal pairs); diff-of-means at raw alpha=8 ~ 14
(norm-unconstrained). Unit-norm injections at delta=150 are on the same scale as a
single-feature clamp at s=150 (decoder columns are unit norm).

Usage (from sae-for-vlm/):
  python analysis/steer_cbar_direction.py
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

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
PAIRS = {
    "push_lr": ("Pushing [something] from left to right",
                "Pushing [something] from right to left"),
    "cam_lr":  ("Turning the camera left while filming [something]",
                "Turning the camera right while filming [something]"),
    "move_ud": ("Moving [something] up", "Moving [something] down"),
    "cam_ud":  ("Turning the camera upwards while filming [something]",
                "Turning the camera downwards while filming [something]"),
    "fall_speed": ("[Something] falling like a rock",
                   "[Something] falling like a feather or paper"),
    "spin_stop": ("Spinning [something] so it continues spinning",
                  "Spinning [something] that quickly stops spinning"),
}
MATCH = {"vel_x": ["push_lr", "cam_lr"], "vel_y": ["move_ud", "cam_ud"],
         "speed": ["fall_speed"], "accel_mag": ["spin_stop"], "direction": ["cam_lr"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--ball_acts", default="local_runs/nfp_results/ball_raw_acts.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--class_speed", default="local_runs/steering/class_speed.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--deltas", nargs="*", type=float, default=[-150, -50, 50, 150])
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expF1_cbar_steering.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    # ---- 1. c_bar from cached ball activations (CPU) ----
    d = torch.load(args.ball_acts, map_location="cpu")
    ball, tau = d["ball"].float(), d["tau"].float()          # [N,8,768], [N,8,5]
    hc = ball - ball.mean(1, keepdim=True)
    tc = tau - tau.mean(1, keepdim=True)
    c_all = torch.einsum("btd,btk->bdk", hc, tc) / ball.shape[1]   # [N,768,5]
    cbar = c_all.mean(0)                                            # [768,5]
    cbar_t = {name: cbar[:, k] for k, name in enumerate(TAU)}
    print("c_bar norms: " + "  ".join(f"{n}={v.norm():.3f}" for n, v in cbar_t.items()))
    U = torch.stack([v / v.norm() for v in cbar_t.values()], 1)     # [768,5]
    cosm = (U.T @ U).numpy()
    print("pairwise cosines among c_bar directions:")
    for i, n in enumerate(TAU):
        print(f"  {n:<10}" + "".join(f"{cosm[i, j]:+8.2f}" for j in range(5)))

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    W = clf.classifier.weight.data.cpu()                            # [174,768]
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy()
    sig = [int(i) for i in np.where((p_all < 0.05 / p_all.shape[0]).any(1))[0]]
    Wd = sae.decoder.weight.data.cpu()[:, sig]
    Wd = Wd / Wd.norm(dim=0, keepdim=True)
    print("\nmax |cos| of each c_bar with the 85 NFP decoder columns:")
    for n, v in cbar_t.items():
        c = (v / v.norm()) @ Wd
        j = int(c.abs().argmax())
        print(f"  {n:<10} max|cos|={float(c.abs().max()):.3f} (feat{sig[j]:05d})")

    head_diff = {}
    print("\ncos(c_bar, head-diff) for matched pairs:")
    for key, (pl, nl) in PAIRS.items():
        hd = W[label2idx[pl]] - W[label2idx[nl]]
        head_diff[key] = hd / hd.norm()
        for n in TAU:
            if key in MATCH.get(n, []):
                cc = float((cbar_t[n] / cbar_t[n].norm()) @ head_diff[key])
                print(f"  {key:<11} vs c_bar[{n:<9}]  cos={cc:+.3f}")

    # ---- 2. steering ----
    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    def probs(cache, vec=None):
        outs = []
        for pv in cache:
            steer.add_vec = vec
            with torch.no_grad():
                outs.append(torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu())
            steer.add_vec = None
        return torch.cat(outs, 0).numpy()

    rngs = [torch.Generator().manual_seed(s) for s in (11, 22)]
    rand_dirs = [torch.randn(768, generator=g) for g in rngs]
    rand_dirs = [r / r.norm() for r in rand_dirs]

    results = {"cbar_norms": {n: float(v.norm()) for n, v in cbar_t.items()},
               "pairs": {}}
    for key, (pl, nl) in PAIRS.items():
        cp, cn = label2idx.get(pl, -1), label2idx.get(nl, -1)
        if cp < 0 or cn < 0:
            continue
        caches = {"pos": cache_of(by_tmpl[pl][: args.per_class]),
                  "neg": cache_of(by_tmpl[nl][: args.per_class])}
        matched = [n for n in TAU if key in MATCH.get(n, [])]
        dirs = [(f"cbar[{n}]", cbar_t[n] / cbar_t[n].norm()) for n in matched]
        dirs += [("head-diff", head_diff[key]),
                 ("rand-11", rand_dirs[0]), ("rand-22", rand_dirs[1])]
        print(f"\n=== {key} ===")
        rec = {}
        for dname, u in dirs:
            los = {}
            for dl_ in args.deltas:
                vec = (dl_ * u).float()
                lo_sides = []
                for side, own, other in [("pos", cp, cn), ("neg", cn, cp)]:
                    P = probs(caches[side], vec)
                    lo_sides.append(float(np.mean(
                        np.log(P[:, cp] + 1e-12) - np.log(P[:, cn] + 1e-12))))
                los[dl_] = lo_sides
            hi, lo_ = max(args.deltas), min(args.deltas)
            shift = float(np.mean(los[hi]) - np.mean(los[lo_])) / 2
            # flips at best orientation: steer neg-side videos toward pos with the
            # better-signed delta, and vice versa
            s_pos = hi if shift > 0 else lo_
            s_neg = lo_ if shift > 0 else hi
            Pn = probs(caches["neg"], (s_pos * u).float())
            Pp = probs(caches["pos"], (s_neg * u).float())
            flip = (float((Pn[:, cp] > Pn[:, cn]).mean())
                    + float((Pp[:, cn] > Pp[:, cp]).mean())) / 2
            top1 = (float((Pn.argmax(1) == cp).mean())
                    + float((Pp.argmax(1) == cn).mean())) / 2
            rec[dname] = {"shift": round(shift, 3), "flip": round(flip, 3),
                          "top1": round(top1, 3)}
            print(f"  {dname:<16} shift={shift:+7.2f}  flip={flip:.2f}  top1={top1:.2f}")
        results["pairs"][key] = rec

    # ---- 3. E[speed] dose-response for cbar[speed] on neutral videos ----
    cs = json.load(open(args.class_speed))["by_idx"]
    speed_z = np.zeros(174)
    for c, v in cs.items():
        speed_z[int(c)] = v["speed_z"]
    cam = {v for v in label2idx if v.startswith("Turning the camera")}
    items = [it for it in val if it.get("template", "") not in cam][:24]
    cache = cache_of(items)
    u = (cbar_t["speed"] / cbar_t["speed"].norm())
    curve = {}
    for dl_ in [-150, -50, 0, 50, 150]:
        vec = None if dl_ == 0 else (dl_ * u).float()
        P = probs(cache, vec)
        curve[dl_] = round(float((P @ speed_z).mean()), 3)
    results["espeed_curve_cbar_speed"] = curve
    print(f"\nE[speed] vs delta for unit c_bar[speed]: {curve}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
