"""
Experiment D1 — playback-speed mediation: what fraction of the model's response to a
REAL change in motion speed flows through the 85 NFP features?

Input-level temporal counterfactual with content held constant: two versions of the
same video that differ only in playback speed.
  1x = 16 frames sampled uniformly over the MIDDLE HALF of the clip.
  2x = 16 frames sampled uniformly over the WHOLE clip (double the frame stride, same
       center; apparent motion speed doubles).
Videos with fewer than 40 decoded frames are skipped so both windows are well defined.

Three measurements:
  (a) Feature response: mean-pooled activation change of each NFP feature from 1x to
      2x. Speed-tagged features with positive NFP t should increase.
  (b) Total effect (TE): the change in the model's predicted motion level,
      E[speed] = sum_c P(c) * speed_z(c) (optical-flow class scores, class_speed.json),
      between the 1x and 2x runs. TE_i per video, averaged.
  (c) Mediation (NIE): run the 1x input but patch the chosen features' token-level
      activations from the 2x run (SteerLayer patch mode). NIE = E[speed](patched) -
      E[speed](1x). Fraction mediated = mean NIE / mean TE.
      Sets: NFP-85, flippers-12, random-85, static-85, ALL-6144 (ceiling).

Caveat logged with the design (arXiv 2606.27510): with multiple interacting mediators,
NIE fractions need not sum to 1 across disjoint sets; the random/static controls and the
ceiling calibrate the scale.

Replication: seed 0; videos = first 300 clips with >= 40 frames from the seed-1-shuffled
validation list; model MCG-NJU/videomae-base-finetuned-ssv2; SAE local_runs/sae/ae.pt;
NFP flags Bonferroni 0.05/6144; static pool finite-t max|t| < 2.

Usage (from sae-for-vlm/):
  python analysis/steer_speed_mediation.py --n_videos 300
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

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


class TwoSpeedFrames(Dataset):
    """Returns (imgs_1x, imgs_2x, id): same clip, middle-half vs full-span sampling."""
    def __init__(self, videos_dir, items):
        self.items = items
        self.dir = Path(videos_dir)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import av
        it = self.items[i]
        try:
            cont = av.open(str(self.dir / f"{it['id']}.webm"))
            frames = [f.to_ndarray(format="rgb24") for f in cont.decode(video=0)]
            cont.close()
            L = len(frames)
            if L < 40:
                raise ValueError("short")
            full = np.linspace(0, L - 1, 16).astype(int)                 # 2x speed
            half = np.linspace(L * 0.25, L * 0.75 - 1, 16).astype(int)   # 1x speed
            im1 = [Image.fromarray(frames[j]) for j in half]
            im2 = [Image.fromarray(frames[j]) for j in full]
        except Exception:
            im1 = im2 = [Image.new("RGB", (224, 224)) for _ in range(16)]
        return im1, im2, it["id"]


def collate_two(proc):
    def c(batch):
        a = proc(images=[b[0] for b in batch], return_tensors="pt")["pixel_values"]
        b_ = proc(images=[b[1] for b in batch], return_tensors="pt")["pixel_values"]
        return a, b_, [x[2] for x in batch]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--class_speed", default="local_runs/steering/class_speed.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_videos", default=300, type=int)
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=6, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD1_speed_mediation.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    cs = json.load(open(args.class_speed))["by_idx"]
    speed_z = np.zeros(len(clf.config.id2label))
    for c, v in cs.items():
        speed_z[int(c)] = v["speed_z"]

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy(); t_all = nfp["t_stat"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = [int(i) for i in np.where((p_all < bonf).any(1))[0]]
    dom = {i: TAU[int(np.argmax(np.abs(t_all[i])))] for i in sig}
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
            ("static-85", torch.tensor(sorted(rng.choice(static_pool, 85, replace=False)))),
            ("ALL-6144", torch.arange(sae.dict_size))]

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    ds = TwoSpeedFrames(args.ssv2_videos, val[: args.n_videos * 2])  # extra; shorts skipped
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=collate_two(proc))

    Es = {"1x": [], "2x": [], **{n: [] for n, _ in sets}}
    A1_sum = np.zeros(sae.dict_size); A2_sum = np.zeros(sae.dict_size); n_done = 0

    for pv1, pv2, ids in dl:
        if n_done >= args.n_videos:
            break
        # 2x pass: probs + full feature capture + mean-pooled acts
        steer.record_tokens_idx = torch.arange(sae.dict_size)
        steer.captured_tokens = []
        steer.record = True; steer.captured = []
        with torch.no_grad():
            P2 = torch.softmax(clf(pixel_values=pv2.to(device)).logits, -1).cpu().numpy()
        f2 = steer.captured_tokens[0]
        A2 = torch.cat(steer.captured, 0).numpy()
        steer.record_tokens_idx = None; steer.record = False; steer.captured = []
        # 1x pass: probs + mean-pooled acts
        steer.record = True
        with torch.no_grad():
            P1 = torch.softmax(clf(pixel_values=pv1.to(device)).logits, -1).cpu().numpy()
        A1 = torch.cat(steer.captured, 0).numpy()
        steer.record = False; steer.captured = []

        Es["1x"] += (P1 @ speed_z).tolist()
        Es["2x"] += (P2 @ speed_z).tolist()
        A1_sum += A1.sum(0); A2_sum += A2.sum(0)
        # mediation passes: 1x input, features patched from the 2x run
        for name, idx in sets:
            steer.patch_idx = idx
            steer.patch_vals = f2[..., idx] if idx.numel() < sae.dict_size else f2
            with torch.no_grad():
                Pm = torch.softmax(clf(pixel_values=pv1.to(device)).logits, -1).cpu().numpy()
            steer.patch_idx = steer.patch_vals = None
            Es[name] += (Pm @ speed_z).tolist()
        del f2
        n_done += pv1.shape[0]
        if n_done % 60 < args.batch_size:
            print(f"  {n_done} videos")

    e1, e2 = np.array(Es["1x"]), np.array(Es["2x"])
    TE = e2 - e1
    print(f"\n=== total effect of 2x playback on E[speed]: {TE.mean():+.3f} "
          f"(se {TE.std()/np.sqrt(len(TE)):.3f}, n={len(TE)}) ===")
    res = {"n": len(TE), "TE_mean": round(float(TE.mean()), 4),
           "TE_se": round(float(TE.std() / np.sqrt(len(TE))), 4), "sets": {}}
    for name, _ in sets:
        NIE = np.array(Es[name]) - e1
        frac = NIE.mean() / (TE.mean() + 1e-9)
        res["sets"][name] = {"NIE_mean": round(float(NIE.mean()), 4),
                             "NIE_se": round(float(NIE.std() / np.sqrt(len(NIE))), 4),
                             "frac_mediated": round(float(frac), 3)}
        print(f"  {name:<12} NIE = {NIE.mean():+.3f} (se {NIE.std()/np.sqrt(len(NIE)):.3f})"
              f"   fraction mediated = {100*frac:.0f}%")

    # feature response by tag
    dA = (A2_sum - A1_sum) / max(n_done, 1)
    print("\n=== mean activation change 1x -> 2x by tag (positive = rises with speed) ===")
    res["act_change"] = {}
    for t in TAU:
        ks = [k for k in sig if dom[k] == t]
        if ks:
            v = float(np.mean(dA[ks]))
            res["act_change"][t] = round(v, 4)
            print(f"  {t:<10} n={len(ks):3d}  mean dA = {v:+.4f}")
    stat_ks = static_pool[:200]
    res["act_change"]["static_pool"] = round(float(np.mean(dA[stat_ks])), 4)
    print(f"  {'static':<10} n={len(stat_ks):3d}  mean dA = {np.mean(dA[stat_ks]):+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
