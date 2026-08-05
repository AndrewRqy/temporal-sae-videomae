"""
Experiment D5 — full-dictionary causal scan by attribution patching: measured NFP recall.

All causal tests so far examined the 85 NFP-flagged features. This asks the converse:
how many causally-temporal features exist in the WHOLE dictionary, and what fraction did
NFP find? Exact interchange patching of all 6,144 features is too expensive; attribution
patching (AtP; arXiv 2403.00745) approximates each feature's interchange effect with one
gradient: for receiver video r with donor d (opposite direction class),

  attr_k ~= sum_tokens dLoss/df_k |_r * (f_d[token,k] - f_r[token,k]),
  Loss = logit(donor class) - logit(own class)  (the pair objective used throughout).

Implementation: layer 11 output is re-expressed as decode(f) + e with f a leaf tensor
requiring grad (AttrLayer below); one backward gives dLoss/df for all features at once.
Donor activations are captured in a preceding no-grad pass, receiver-donor pairing i-i
as in expC1.

Protocol: 7 temporal pairs x both directions x 12 videos/side, batch 4. Per pair the
per-feature attributions are summed over tokens and videos; the global score is the mean
of |attr| across pairs, each pair normalized by its own max |attr| so no single pair
dominates. Outputs:
  - top-attributed feature list with NFP membership and step-14 axis membership;
  - NFP recall metrics: how many NFP-85 land in the top-85 / top-200 attributed;
    AUROC of NFP membership against the attribution score; median rank of NFP features;
  - validation: exact single-feature interchange effects for the top-15 attributed
    non-NFP features and 15 NFP features on cam_lr; correlation exact vs attribution.

Replication: seed 0; videos identical to expC1 (first 12/class, seed-1-shuffled val);
model MCG-NJU/videomae-base-finetuned-ssv2; SAE local_runs/sae/ae.pt.

Usage (from sae-for-vlm/):
  python analysis/attr_temporal_scan.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from analysis.steer_ssv2_logits import ssv2_collate
from analysis.steer_pair_screen import ItemFrames

TAU = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
PAIRS7 = [
    ("push_lr", "Pushing [something] from left to right", "Pushing [something] from right to left"),
    ("pull_lr", "Pulling [something] from left to right", "Pulling [something] from right to left"),
    ("cam_lr",  "Turning the camera left while filming [something]",
                "Turning the camera right while filming [something]"),
    ("move_ud", "Moving [something] up", "Moving [something] down"),
    ("cam_ud",  "Turning the camera upwards while filming [something]",
                "Turning the camera downwards while filming [something]"),
    ("fall_speed", "[Something] falling like a rock", "[Something] falling like a feather or paper"),
    ("spin_stop", "Spinning [something] so it continues spinning",
                  "Spinning [something] that quickly stops spinning"),
]


class AttrLayer(nn.Module):
    """Re-express layer output as decode(f)+e with f a grad leaf; optional capture/patch."""
    def __init__(self, base, sae):
        super().__init__()
        self.base, self.sae = base, sae
        self.mode = "plain"          # plain | capture | grad | patch
        self.f_store = None          # capture: detached f; grad: leaf f
        self.patch = None            # (idx, vals) for exact validation patches

    def forward(self, hidden_states, *args, **kwargs):
        kwargs.pop("head_mask", None)
        out = self.base(hidden_states, *args, **kwargs)
        is_tuple = isinstance(out, tuple)
        acts = out[0] if is_tuple else out
        rest = out[1:] if is_tuple else ()
        if self.mode == "capture":
            with torch.no_grad():
                self.f_store = self.sae.encode(acts).detach()
        elif self.mode == "grad":
            with torch.no_grad():
                f0 = self.sae.encode(acts)
                e = acts - self.sae.decode(f0)
            f = f0.detach().requires_grad_(True)
            self.f_store = f
            acts = self.sae.decode(f) + e.detach()
        elif self.mode == "patch" and self.patch is not None:
            with torch.no_grad():
                f = self.sae.encode(acts)
                e = acts - self.sae.decode(f)
                f = f.clone()
                idx, vals = self.patch
                f[..., idx] = vals.to(f.dtype).to(f.device)
                acts = self.sae.decode(f) + e
        return ((acts,) + rest) if is_tuple else acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base-finetuned-ssv2")
    ap.add_argument("--sae_path", default="local_runs/sae/ae.pt")
    ap.add_argument("--nfp_results", default="local_runs/nfp_results/sae_nfp.pt")
    ap.add_argument("--axis_json", default="local_runs/steering/expB_axis_decomposition.json")
    ap.add_argument("--ssv2_videos", default="../SSv2/videos")
    ap.add_argument("--ssv2_val_json",
                    default="../SSv2/raw/20bn-something-something-download-package-labels/labels/validation.json")
    ap.add_argument("--layer", default=11, type=int)
    ap.add_argument("--per_class", default=12, type=int)
    ap.add_argument("--batch_size", default=4, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="local_runs/steering/expD5_attr_scan.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    clf = VideoMAEForVideoClassification.from_pretrained(args.model_name).to(device).eval()
    label2idx = {v: int(k) for k, v in clf.config.id2label.items()}
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device); sae.eval()
    attr_layer = AttrLayer(clf.videomae.encoder.layer[args.layer], sae).to(device)
    clf.videomae.encoder.layer[args.layer] = attr_layer
    proc = VideoMAEImageProcessor.from_pretrained(args.model_name)

    nfp = torch.load(args.nfp_results, map_location="cpu")
    p_all = nfp["p_val"].numpy()
    bonf = 0.05 / p_all.shape[0]
    sig = set(int(i) for i in np.where((p_all < bonf).any(1))[0])
    axis = sorted({f["idx"] for pr in json.load(open(args.axis_json))["pairs"]
                   for f in pr["top10"]})

    val = json.load(open(args.ssv2_val_json))
    vrng = np.random.RandomState(args.seed + 1); vrng.shuffle(val)
    by_tmpl = {}
    for it in val:
        by_tmpl.setdefault(it.get("template", ""), []).append(it)

    def cache_of(items):
        dl = DataLoader(ItemFrames(args.ssv2_videos, items), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=ssv2_collate(proc))
        return [b[0]["pixel_values"] for b in dl]

    D = sae.dict_size
    attr_by_pair = {}
    for key, pos_l, neg_l in PAIRS7:
        cp, cn = label2idx.get(pos_l, -1), label2idx.get(neg_l, -1)
        if cp < 0 or cn < 0 or min(len(by_tmpl.get(pos_l, [])), len(by_tmpl.get(neg_l, []))) < args.per_class:
            continue
        caches = {"pos": cache_of(by_tmpl[pos_l][: args.per_class]),
                  "neg": cache_of(by_tmpl[neg_l][: args.per_class])}
        acc = torch.zeros(D)
        for recv, donor, own, tgt in [("pos", "neg", cp, cn), ("neg", "pos", cn, cp)]:
            for pv_r, pv_d in zip(caches[recv], caches[donor]):
                # donor capture (no grad)
                attr_layer.mode = "capture"
                with torch.no_grad():
                    clf(pixel_values=pv_d.to(device))
                f_d = attr_layer.f_store
                # receiver pass with grad on f
                attr_layer.mode = "grad"
                logits = clf(pixel_values=pv_r.to(device)).logits
                loss = (logits[:, tgt] - logits[:, own]).mean()
                loss.backward()
                f_r = attr_layer.f_store
                a = (f_r.grad * (f_d - f_r.detach())).sum(dim=(0, 1)).detach().cpu()
                acc += a
                attr_layer.mode = "plain"; attr_layer.f_store = None
        attr_by_pair[key] = acc.numpy()
        top = np.argsort(-np.abs(acc.numpy()))[:5]
        print(f"  {key:<11} top attributed: "
              + ", ".join(f"feat{int(k):05d}{'*' if int(k) in sig else ''}" for k in top))

    # global score: mean of per-pair |attr| normalized by pair max
    A = np.stack([np.abs(v) / (np.abs(v).max() + 1e-12) for v in attr_by_pair.values()])
    score = A.mean(0)
    order = np.argsort(-score)

    in_sig = np.array([k in sig for k in range(D)])
    top85 = set(order[:85].tolist()); top200 = set(order[:200].tolist())
    r85 = len(top85 & sig); r200 = len(top200 & sig)
    ranks = np.empty(D); ranks[order] = np.arange(D)
    med_rank_sig = float(np.median(ranks[list(sig)]))
    # AUROC of NFP membership vs score
    from scipy import stats as st
    auroc = st.mannwhitneyu(score[in_sig], score[~in_sig], alternative="greater")
    U = auroc.statistic / (in_sig.sum() * (~in_sig).sum())
    ax_in_top200 = len(set(axis) & top200)
    print(f"\n=== NFP recall against the attribution ranking ===")
    print(f"  NFP-85 in top-85 attributed : {r85}/85")
    print(f"  NFP-85 in top-200 attributed: {r200}/85")
    print(f"  median attribution rank of NFP features: {med_rank_sig:.0f} / {D}")
    print(f"  AUROC (NFP membership vs score): {U:.3f}  (MW p={auroc.pvalue:.2e})")
    print(f"  step-14 axis features in top-200: {ax_in_top200}/{len(axis)}")
    print(f"\n  top-25 attributed features overall:")
    for r, k in enumerate(order[:25]):
        k = int(k)
        print(f"    #{r+1:3d} feat{k:05d}  score={score[k]:.3f}  "
              f"{'NFP' if k in sig else ''}{' AXIS' if k in axis else ''}")

    # exact validation on cam_lr: top-15 attributed non-NFP + 15 NFP by attribution
    key, pos_l, neg_l = PAIRS7[2]
    cp, cn = label2idx[pos_l], label2idx[neg_l]
    caches = {"pos": cache_of(by_tmpl[pos_l][: args.per_class]),
              "neg": cache_of(by_tmpl[neg_l][: args.per_class])}
    val_feats = [int(k) for k in order if int(k) not in sig][:15] + \
                sorted(sig, key=lambda k: -score[k])[:15]
    attr_layer.mode = "capture"
    donor_f = []
    with torch.no_grad():
        for pv in caches["neg"]:
            clf(pixel_values=pv.to(device))
            donor_f.append(attr_layer.f_store.cpu())
    attr_layer.mode = "plain"

    def pair_lo(cache, patch=None):
        outs = []
        for bi, pv in enumerate(cache):
            if patch is not None:
                attr_layer.mode = "patch"
                attr_layer.patch = (patch[0], patch[1][bi])
            with torch.no_grad():
                P = torch.softmax(clf(pixel_values=pv.to(device)).logits, -1).cpu().numpy()
            attr_layer.mode = "plain"; attr_layer.patch = None
            outs.append(np.log(P[:, cn] + 1e-12) - np.log(P[:, cp] + 1e-12))
        return float(np.concatenate(outs).mean())

    base = pair_lo(caches["pos"])
    exact, approx = [], []
    for k in val_feats:
        idx = torch.tensor([k])
        vals = [df[..., idx] for df in donor_f]
        d = pair_lo(caches["pos"], patch=(idx, vals)) - base
        exact.append(d); approx.append(float(score[k]))
    rho = st.spearmanr(exact, approx)
    print(f"\n  exact-vs-attribution validation (cam_lr, 30 features): "
          f"spearman rho={rho.statistic:.2f} (p={rho.pvalue:.3f})")

    res = {"recall_top85": r85, "recall_top200": r200,
           "median_rank_nfp": med_rank_sig, "auroc": round(float(U), 3),
           "axis_in_top200": ax_in_top200,
           "top100": [{"idx": int(k), "score": round(float(score[k]), 4),
                       "nfp": bool(int(k) in sig), "axis": bool(int(k) in axis)}
                      for k in order[:100]],
           "validation_spearman": round(float(rho.statistic), 3)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
