"""
Test whether top nonmono speed-responsive SAE features transfer to the
velocity and acceleration datasets.

Correlation method differs by dataset type because the speed profile differs:

  velocity     Ball moves at CONSTANT speed for all 16 frames.
               �?No within-video temporal speed variation.
               �?Cross-video test: does feature's mean activation (avg over
                 8 time steps) correlate with the video's speed_mps?

  acceleration Ball ACCELERATES from rest: v(t) �?a × t_sec.
               �?Speed increases linearly across the 8 tubelets.
               �?Within-video test: does feature's 8-step trajectory
                 correlate with the expected linear speed ramp?
               �?Cross-video: does the slope of that ramp correlate with
                 the video's accel_mps2?

Video selection (all available videos):
  velocity    : 8 dirs x 7 speeds x 7 positions  = 392 videos
  acceleration: 8 dirs x 5 accels x 7 positions  = 280 videos

Cross-reference with nonmono results:
  Pass --nonmono_pkl to load top nonmono features and check their rank here.

Output .pkl:
  {
    "dataset_type":   str,
    "features":       np.ndarray [N, 8, dict_size],
    "labels":         list[dict],
    "video_ids":      list[str],
    "speed_ref":      list[np.ndarray [8]],  # per-video expected speed (m/s)
    # velocity-specific
    "crossvid_corr":  np.ndarray [dict_size],  # r(mean_act, speed) across videos
    # acceleration-specific
    "temporal_corr":  np.ndarray [dict_size],  # avg within-video r(traj, ramp)
    "slope_corr":     np.ndarray [dict_size],  # r(traj_slope, accel) across videos
    "top_k_indices":  np.ndarray [K],
    "sae_type":       str,
    "sae_path":       str,
  }

Usage:
  python analyze_vel_accel_temporal.py \\
    --dataset_type  velocity \\
    --dataset_dir   /path/to/ball_dataset/velocity \\
    --sae_path      /path/to/ae.pt \\
    --sae_type      batch_top_k \\
    --output_path   vel_btk.pkl \\
    --nonmono_pkl   /path/to/nonmono_btk.pkl

  python analyze_vel_accel_temporal.py \\
    --dataset_type  acceleration \\
    --dataset_dir   /path/to/ball_dataset/acceleration \\
    --sae_path      /path/to/ae.pt \\
    --sae_type      standard \\
    --output_path   accel_std.pkl \\
    --nonmono_pkl   /path/to/nonmono_std.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers import BatchTopKSAE
from models.videomae import VideoMAE

# VideoMAE token layout
N_TEMPORAL = 8     # 16 frames / 2 per tubelet
N_SPATIAL  = 196   # 14×14 spatial patches
FRAME_RATE = 24
STEP_RATE  = 240   # 10 physics substeps per frame


# ---------------------------------------------------------------------------
# Video selection
# ---------------------------------------------------------------------------
def select_videos(dataset_dir: Path, dataset_type: str) -> list[Path]:
    """
    Return all available video dirs from the dataset.

    velocity    : 8 dirs x 7 speeds x 7 positions = 392 videos
    acceleration: 8 dirs x 5 accels x 7 positions = 280 videos
    """
    if dataset_type == "velocity":
        n_params, n_dirs, n_pos = 7, 8, 7
    elif dataset_type == "acceleration":
        n_params, n_dirs, n_pos = 5, 8, 7
    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'")

    dirs = []
    for d in range(n_dirs):
        for p in range(n_params):
            for s in range(n_pos):
                path = dataset_dir / f"dir{d:02d}_param{p:02d}_pos{s:02d}"
                if path.exists():
                    dirs.append(path)

    if not dirs:
        raise FileNotFoundError(
            f"No matching video folders found in {dataset_dir}. "
            "Check --dataset_dir and --dataset_type.")

    print(f"Selected {len(dirs)} videos from {dataset_dir}")
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Expected per-video speed profile (8 tubelet steps)
# ---------------------------------------------------------------------------
def expected_speed_profile(label: dict, dataset_type: str,
                           reverse: bool = False) -> np.ndarray:
    """
    Return the expected speed (m/s) at each of the 8 tubelet time steps.

    velocity     �?flat line at speed_mps  (reverse has no effect)
    acceleration �?linear ramp: v(t) = a × (2t / FRAME_RATE)
    deceleration �?reversed ramp: v(t) = a × (2(7-t) / FRAME_RATE)
                   (same videos, frames played backwards)
    """
    t = np.arange(N_TEMPORAL, dtype=np.float32)

    if dataset_type == "velocity":
        return np.full(N_TEMPORAL, label["speed_mps"], dtype=np.float32)
    else:
        a = label["accel_mps2"]
        ramp = (a * 2 * t / FRAME_RATE).astype(np.float32)
        return ramp[::-1].copy() if reverse else ramp


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VelAccelDataset(Dataset):
    def __init__(self, video_dirs: list[Path], num_frames: int = 16,
                 reverse_frames: bool = False):
        self.video_dirs     = video_dirs
        self.num_frames     = num_frames
        self.reverse_frames = reverse_frames

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        frame_indices = range(self.num_frames)
        if self.reverse_frames:
            frame_indices = range(self.num_frames - 1, -1, -1)
        frames = [
            Image.open(vdir / f"rgba_{i:05d}.png").convert("RGB")
            for i in frame_indices
        ]
        with open(vdir / "metadata.json") as f:
            label = json.load(f)["label"]
        return frames, label, vdir.name


def make_collate(processor):
    def collate(batch):
        frames_list = [item[0] for item in batch]
        labels      = [item[1] for item in batch]
        video_ids   = [item[2] for item in batch]
        inputs = processor(images=frames_list, return_tensors="pt")
        return inputs, labels, video_ids
    return collate


# ---------------------------------------------------------------------------
# SAE loader
# ---------------------------------------------------------------------------
def load_sae(sae_type: str, sae_path: str, device: str):
    if sae_type == "batch_top_k":
        sae = BatchTopKSAE.from_pretrained(sae_path, device=device)
    elif sae_type == "standard":
        sae = AutoEncoder.from_pretrained(sae_path, device=device)
    else:
        raise ValueError(f"Unknown sae_type '{sae_type}'")
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Correlation utilities
# ---------------------------------------------------------------------------
def batch_pearsonr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pearson r between each row of X (shape [N, T]) and y (shape [T]).
    Returns [N] correlations; sets r=0 where X row is constant.
    """
    y_c = y - y.mean()
    X_c = X - X.mean(axis=1, keepdims=True)
    num     = (X_c * y_c[np.newaxis, :]).sum(axis=1)
    denom_X = np.sqrt((X_c ** 2).sum(axis=1))
    denom_y = float(np.sqrt((y_c ** 2).sum()))
    denom   = denom_X * denom_y
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / denom
    r[~np.isfinite(r)] = 0.0
    return r  # [N]


def pearsonr_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar Pearson r between two 1-D arrays; returns 0 if either is constant."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt((a_c**2).sum() * (b_c**2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def linear_slope(y: np.ndarray) -> float:
    """Slope from least-squares linear fit to y vs 0..len(y)-1."""
    x = np.arange(len(y), dtype=np.float32)
    x_c = x - x.mean()
    return float((x_c * (y - y.mean())).sum() / ((x_c**2).sum() + 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_type", required=True,
                   choices=["velocity", "acceleration"])
    p.add_argument("--dataset_dir",  required=True)
    p.add_argument("--sae_path",     required=True)
    p.add_argument("--sae_type",     required=True,
                   choices=["batch_top_k", "standard"])
    p.add_argument("--output_path",  required=True)
    p.add_argument("--nonmono_pkl",  default=None,
                   help="Path to nonmono result pkl for cross-reference")
    p.add_argument("--reverse_frames", action="store_true",
                   help="Reverse frame order (acceleration �?deceleration)")
    p.add_argument("--top_k",        default=50, type=int)
    p.add_argument("--model_name",   default="MCG-NJU/videomae-base-finetuned-ssv2")
    p.add_argument("--layer",        default=11, type=int)
    p.add_argument("--attachment_point", default="post_mlp_residual")
    p.add_argument("--batch_size",   default=4,  type=int)
    p.add_argument("--num_workers",  default=4,  type=int)
    p.add_argument("--device",       default="cuda:0")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device
    dt     = args.dataset_type

    dataset_dir = Path(args.dataset_dir)
    video_dirs  = select_videos(dataset_dir, dt)

    # Optional: load nonmono top features for cross-reference
    nonmono_top = None
    if args.nonmono_pkl:
        with open(args.nonmono_pkl, "rb") as f:
            nm = pickle.load(f)
        nonmono_top = nm["top_k_indices"][:20]
        print(f"Loaded nonmono top features: {nonmono_top[:10].tolist()} �?)

    # ── VideoMAE ──────────────────────────────────────────────────────────
    print(f"Loading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)
    model.attach(args.attachment_point, args.layer, sae=None)
    hook_key = f"{args.attachment_point}_{args.layer}"

    # ── SAE ───────────────────────────────────────────────────────────────
    print(f"Loading SAE ({args.sae_type}): {args.sae_path}")
    sae = load_sae(args.sae_type, args.sae_path, device)

    # ── DataLoader ────────────────────────────────────────────────────────
    if args.reverse_frames:
        print("Frame order: REVERSED (deceleration mode)")

    ds = VelAccelDataset(video_dirs, reverse_frames=args.reverse_frames)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=make_collate(model.processor))

    all_features  = []
    all_labels    = []
    all_video_ids = []

    # ── Extract temporal features (steps 1-3, same as nonmono) ───────────
    print(f"\nExtracting temporal features ({dt}) ...")
    for inputs, labels, video_ids in tqdm(dl):
        model.encode(inputs)
        acts = model.register[hook_key][0]            # [B, 1568, 768]
        B    = acts.shape[0]
        acts = acts.view(B, N_TEMPORAL, N_SPATIAL, -1)
        acts = acts.max(dim=2).values.to(device)      # [B, 8, 768]

        with torch.no_grad():
            flat  = acts.reshape(B * N_TEMPORAL, -1)
            feats = sae.encode(flat).reshape(B, N_TEMPORAL, -1)

        all_features.append(feats.cpu().numpy())
        all_labels.extend(labels)
        all_video_ids.extend(video_ids)

    features = np.concatenate(all_features, axis=0)  # [N, 8, dict_size]
    N, T, D  = features.shape
    print(f"Feature tensor: {features.shape}")

    # ── Build per-video expected speed profiles ────────────────────────────
    speed_refs = np.stack([
        expected_speed_profile(lbl, dt, reverse=args.reverse_frames)
        for lbl in all_labels
    ])  # [N, 8]

    # ── Correlation analysis ───────────────────────────────────────────────
    print(f"\nComputing correlations ({dt}) ...")

    if dt == "velocity":
        # Cross-video: r(feature_mean_activation, speed_mps) across N videos
        feat_mean = features.mean(axis=1)                   # [N, dict_size]
        speed_vals = speed_refs[:, 0]                        # [N] �?all same within a video

        feat_mean_c = feat_mean - feat_mean.mean(axis=0, keepdims=True)
        spd_c       = speed_vals - speed_vals.mean()
        num   = (feat_mean_c * spd_c[:, np.newaxis]).sum(axis=0)  # [dict_size]
        denom = (np.sqrt((feat_mean_c**2).sum(axis=0)) *
                 float(np.sqrt((spd_c**2).sum())))
        with np.errstate(invalid="ignore", divide="ignore"):
            crossvid_corr = num / denom
        crossvid_corr[~np.isfinite(crossvid_corr)] = 0.0    # [dict_size]

        primary_corr = crossvid_corr
        temporal_corr = None
        slope_corr    = None

    else:  # acceleration
        # Within-video: r(feature_traj, expected_speed_ramp) per video, then avg
        temporal_corr_mat = np.zeros((N, D), dtype=np.float32)
        for v in range(N):
            ramp = speed_refs[v]                              # [8], linear ramp
            if ramp.std() < 1e-8:
                continue                                      # zero-accel video, skip
            traj = features[v]                               # [8, D]
            # batch_pearsonr expects [N_samples, T] �?transpose and back
            temporal_corr_mat[v] = batch_pearsonr(traj.T, ramp)  # [D]

        temporal_corr = temporal_corr_mat.mean(axis=0)       # [D]

        # Cross-video: r(linear slope of traj, accel_mps2) across videos
        accel_vals = np.array([lbl["accel_mps2"] for lbl in all_labels],
                              dtype=np.float32)               # [N]
        slopes = np.array([
            [linear_slope(features[v, :, f]) for f in range(D)]
            for v in range(N)
        ], dtype=np.float32)                                  # [N, D]

        slope_c  = slopes - slopes.mean(axis=0, keepdims=True)
        accel_c  = accel_vals - accel_vals.mean()
        num      = (slope_c * accel_c[:, np.newaxis]).sum(axis=0)
        denom    = (np.sqrt((slope_c**2).sum(axis=0)) *
                    float(np.sqrt((accel_c**2).sum())))
        with np.errstate(invalid="ignore", divide="ignore"):
            slope_corr = num / denom
        slope_corr[~np.isfinite(slope_corr)] = 0.0           # [D]

        crossvid_corr = None
        primary_corr  = temporal_corr

    # ── Rankings ──────────────────────────────────────────────────────────
    top_k        = min(args.top_k, D)
    top_k_indices = np.argsort(primary_corr)[::-1][:top_k]

    metric_name = ("cross-video speed r" if dt == "velocity"
                   else "within-video ramp r (mean)")

    print(f"\n{'─'*65}")
    print(f"Top-20 features by {metric_name}  [{dt} / {args.sae_type}]")
    if dt == "velocity":
        print(f"{'rank':>4}  {'feat':>6}  {'cv_speed_r':>11}  {'nonmono_rank':>12}")
    else:
        print(f"{'rank':>4}  {'feat':>6}  {'temporal_r':>11}  {'slope_accel_r':>14}  {'nonmono_rank':>12}")
    print(f"{'─'*65}")

    nm_rank_of = {}
    if nonmono_top is not None:
        nm_rank_of = {int(fi): r+1 for r, fi in enumerate(nonmono_top)}

    for rank, fi in enumerate(top_k_indices[:20], 1):
        nm_r = nm_rank_of.get(int(fi), "-")
        if dt == "velocity":
            print(f"{rank:4d}  {fi:6d}  {crossvid_corr[fi]:11.4f}  {str(nm_r):>12}")
        else:
            print(f"{rank:4d}  {fi:6d}  {temporal_corr[fi]:11.4f}  "
                  f"{slope_corr[fi]:14.4f}  {str(nm_r):>12}")
    print(f"{'─'*65}")

    # Cross-reference: where do nonmono top features rank here?
    if nonmono_top is not None:
        print(f"\nNonmono top features ranked in THIS dataset:")
        rank_order = {int(fi): r+1 for r, fi in enumerate(top_k_indices)}
        print(f"  {'nm_rank':>8}  {'feat':>6}  {'here_rank':>10}  {'primary_r':>10}")
        for nm_r, fi in enumerate(nonmono_top[:10], 1):
            here_r = rank_order.get(int(fi), f">{top_k}")
            print(f"  {nm_r:8d}  {fi:6d}  {str(here_r):>10}  {primary_corr[fi]:10.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "dataset_type":  dt,
        "features":      features,
        "labels":        all_labels,
        "video_ids":     all_video_ids,
        "speed_refs":    speed_refs,
        "top_k_indices": top_k_indices,
        "sae_type":      args.sae_type,
        "sae_path":      args.sae_path,
        "model_name":    args.model_name,
        "layer":         args.layer,
    }
    if dt == "velocity":
        out["crossvid_corr"] = crossvid_corr
    else:
        out["temporal_corr"] = temporal_corr
        out["slope_corr"]    = slope_corr

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved �?{out_path}")


if __name__ == "__main__":
    main()
