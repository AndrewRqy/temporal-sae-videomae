"""
Experiment D4 — SAE-bottleneck probe: is the identified feature set sufficient, as a
representation, for recognizing the temporal concept classes?

Phase 1 (GPU, once): cache per-video mean-pooled layer-11 representations for n videos
per class over all 174 classes: the raw 768-d residual and the 6144-d SAE feature vector.
Phase 2 (CPU): logistic-regression probes on frozen features.

Probe feature sets: NFP-85, flippers-12, random-85, static-85, raw residual 768
(ceiling), all SAE 6144. Tasks:
  (a) 174-way classification (train/test split within each class);
  (b) restricted family task: classification among the C4 concept-family classes only;
  (c) family-vs-rest detection is implied by (a) per-class accuracies.
Also a data-efficiency curve: 3 / 8 / 18 training examples per class.

Known caveat, stated in advance (arXiv 2502.16681; DeepMind SAE-probing negative
results): raw-activation probes are strong baselines and typically win overall. The
claim tested here is narrower: the 85 identified features alone support recognition of
THEIR concept-family classes far above matched random/static feature sets.

Replication: seed 0; 25 videos per class from the seed-1-shuffled validation list;
train = first 18 per class, test = last 7; scikit-learn LogisticRegression
(lbfgs, C=1.0, max_iter=2000, multinomial).

Usage (from sae-for-vlm/):
  python analysis/probe_bottleneck.py --n_per_class 25
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]


def cache_phase(args):
    from torch.utils.data import DataLoader
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    from dictionary_learning import AutoEncoder
    from analysis.steer_ssv2_logits import SteerLayer, ssv2_collate
    from analysis.steer_pair_screen import ItemFrames

    device = torch.device(args.device)
    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    steer = SteerLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = steer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_cls = {}
    for it in val:
        c = label2idx.get(it.get("template", ""), -1)
        if c >= 0:
            by_cls.setdefault(c, []).append(it)
    items, labels = [], []
    for c in sorted(by_cls):
        take = by_cls[c][: args.n_per_class]
        items += take; labels += [c] * len(take)

    dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                    shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
    F, R = [], []
    steer.enabled = False
    steer.record = True; steer.record_raw = True
    done = 0
    for b in dl:
        steer.captured = []; steer.captured_raw = []
        with torch.no_grad():
            clf(pixel_values=b[0]["pixel_values"].to(device))
        F.append(torch.cat(steer.captured, 0))
        R.append(torch.cat(steer.captured_raw, 0))
        done += b[0]["pixel_values"].shape[0]
        if done % 400 < args.batch_size:
            print(f"  cached {done}/{len(items)}")
    torch.save({"feats": torch.cat(F, 0), "resid": torch.cat(R, 0),
                "labels": torch.tensor(labels)}, args.cache)
    print(f"cached {len(items)} videos -> {args.cache}")


def probe_phase(args):
    from sklearn.linear_model import LogisticRegression

    d = torch.load(args.cache, map_location="cpu")
    X_f = d["feats"].numpy(); X_r = d["resid"].numpy(); y = d["labels"].numpy()

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy(); t_all = nfp["t_stat"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = sorted(int(i) for i in np.where((p_all < bonf).any(1))[0])
    finite = np.isfinite(t_all).all(1)
    low_t = (np.abs(np.nan_to_num(t_all, nan=1e9)).max(1) < args.static_t_bar)
    static_pool = [int(i) for i in np.where(finite & low_t)[0] if i not in set(sig)]
    screen = json.load(open(args.screen_json))
    flippers = sorted(r["idx"] for r in screen["features"] if r["n_pairs_flip50"] >= 1)
    rng = np.random.RandomState(args.seed + 7)
    rnd85 = sorted(rng.choice([k for k in range(X_f.shape[1]) if k not in set(sig)],
                              85, replace=False))
    stat85 = sorted(rng.choice(static_pool, 85, replace=False))
    fam = [r["cls"] for r in json.load(open(args.c4_json))["per_class_top"]
           if r["excess"] >= 0.4]

    featsets = [("NFP-85", X_f[:, sig]), ("flippers-12", X_f[:, flippers]),
                ("random-85", X_f[:, rnd85]), ("static-85", X_f[:, stat85]),
                ("resid-768", X_r), ("SAE-6144", X_f)]

    # per-class train/test split: first n_train of each class train, rest test
    def split(y, n_train):
        tr = np.zeros(len(y), bool)
        for c in np.unique(y):
            idxs = np.where(y == c)[0]
            tr[idxs[:n_train]] = True
        return tr

    res = {"tasks": {}}
    for task, mask in [("all-174", np.ones(len(y), bool)),
                       ("family-only", np.isin(y, fam))]:
        ym = y[mask]
        res["tasks"][task] = {}
        print(f"\n=== task: {task} ({mask.sum()} videos, {len(np.unique(ym))} classes) ===")
        for n_train in args.train_sizes:
            tr = split(ym, n_train)
            row = {}
            for name, X in featsets:
                Xm = X[mask]
                clfp = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
                clfp.fit(Xm[tr], ym[tr])
                acc = float(clfp.score(Xm[~tr], ym[~tr]))
                row[name] = round(acc, 3)
            res["tasks"][task][f"train{n_train}"] = row
            print(f"  train={n_train:<3} " + "  ".join(f"{n}:{row[n]:.3f}" for n, _ in featsets))

    # per-class accuracy on family classes with NFP-85 at max train size (from all-174 probe)
    tr = split(y, args.train_sizes[-1])
    clfp = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clfp.fit(X_f[tr][:, sig], y[tr])
    pred = clfp.predict(X_f[~tr][:, sig]); yt = y[~tr]
    fam_acc = float((pred[np.isin(yt, fam)] == yt[np.isin(yt, fam)]).mean())
    non_acc = float((pred[~np.isin(yt, fam)] == yt[~np.isin(yt, fam)]).mean())
    res["nfp85_all174_family_acc"] = round(fam_acc, 3)
    res["nfp85_all174_nonfamily_acc"] = round(non_acc, 3)
    print(f"\nNFP-85 probe (174-way): family-class acc {fam_acc:.3f} vs "
          f"non-family {non_acc:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--screen_json", default="local_runs/steering/expB2_pair_screen.json")
    ap.add_argument("--c4_json", default="local_runs/steering/expC4_span_erasure.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--n_per_class", default=25, type=int)
    ap.add_argument("--train_sizes", nargs="*", type=int, default=[3, 8, 18])
    ap.add_argument("--static_t_bar", default=2.0, type=float)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default="local_runs/steering/expD4_probe_cache.pt")
    ap.add_argument("--out", default="local_runs/steering/expD4_bottleneck_probe.json")
    ap.add_argument("--phase", default="both", choices=["cache", "probe", "both"])
    args = ap.parse_args()
    if args.phase in ("cache", "both") and not Path(args.cache).exists():
        cache_phase(args)
    if args.phase in ("probe", "both"):
        probe_phase(args)


if __name__ == "__main__":
    main()
