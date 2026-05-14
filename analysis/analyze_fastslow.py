"""
Temporal SAE feature analysis for the fast-slow-fast ball dataset.

Pipeline (5 steps):
  1. VideoMAE forward pass         �?[1568, 768] tokens per video
  2. SAE encode each token         �?[1568, dict_size] sparse activations
  3. Temporal aggregation          �?[8, dict_size]
       reshape [1568, 768] �?[8 time, 196 spatial, 768]
       max pool over 196 spatial   �?[8, dict_size]
  4. Speed correlation             per feature per video:
       correlate 8-step trajectory with known tubelet-level speed profile
  5. Consistency check             rank by mean correlation across all N videos

The tubelet speed reference is derived from the dataset's frame-level speed profile
(frames 0-4 fast, 5-10 slow, 11-15 fast) averaged over the 2 frames per tubelet:
  t=0: (fast+fast)/2 = fast   (6.0)
  t=1: (fast+fast)/2 = fast   (6.0)
  t=2: (fast+slow)/2 = mixed  (3.75)
  t=3: (slow+slow)/2 = slow   (1.5)
  t=4: (slow+slow)/2 = slow   (1.5)
  t=5: (slow+fast)/2 = mixed  (3.75)
  t=6: (fast+fast)/2 = fast   (6.0)
  t=7: (fast+fast)/2 = fast   (6.0)

Output .pkl:
  {
    "features":        np.ndarray [N, 8, dict_size],
    "labels":          list[dict],
    "video_ids":       list[str],
    "speed_profile":   np.ndarray [8],           # tubelet-level reference (m/s)
    "corr_per_video":  np.ndarray [N, dict_size],# Pearson r per (video, feature)
    "corr_mean":       np.ndarray [dict_size],
    "corr_std":        np.ndarray [dict_size],
    "corr_frac_pos":   np.ndarray [dict_size],   # fraction of videos with r > 0
    "top_k_indices":   np.ndarray [K],           # top features by mean corr
    "sae_type":        str,
    "sae_path":        str,
  }

Usage:
  python analyze_fastslow_temporal.py \\
    --dataset_dir /path/to/fastslow \\
    --sae_path    /path/to/ae.pt \\
    --sae_type    batch_top_k \\
    --output_path fastslow_btk.pkl \\
    --top_k       50
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

# VideoMAE tubelet structure
N_TEMPORAL = 8    # 16 frames / 2 frames per tubelet
N_SPATIAL  = 196  # 14×14 patches per frame
N_TOKENS   = N_TEMPORAL * N_SPATIAL  # 1568

# Fast-slow-fast speed profile (m/s per frame, matching fastslow_ball_dataset.py)
SLOW_SPEED = 1.5
FAST_SPEED = 6.0
_FRAME_SPEEDS = [FAST_SPEED]*5 + [SLOW_SPEED]*6 + [FAST_SPEED]*5  # 16 values


def compute_tubelet_speed_profile() -> np.ndarray:
    """
    Average adjacent frame-level speeds into 8 tubelet-level speeds.
    Tubelet t covers frames 2t and 2t+1.
    """
    profile = np.array([
        (_FRAME_SPEEDS[2*t] + _FRAME_SPEEDS[2*t + 1]) / 2.0
        for t in range(N_TEMPORAL)
    ])
    return profile   # shape [8]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FastSlowDataset(Dataset):
    """Loads all dir{d:02d}_pos{p:02d}/ videos from a fastslow dataset directory."""

    def __init__(self, dataset_dir: str, num_frames: int = 16):
        self.num_frames = num_frames
        root = Path(dataset_dir)
        self.video_dirs = sorted([
            d for d in root.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])
        if not self.video_dirs:
            raise FileNotFoundError(f"No video folders found in {dataset_dir}")
        print(f"Found {len(self.video_dirs)} videos in {dataset_dir}")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        frames = [
            Image.open(vdir / f"rgba_{i:05d}.png").convert("RGB")
            for i in range(self.num_frames)
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
        raise ValueError(f"Unknown sae_type '{sae_type}'. Use: batch_top_k | standard")
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Vectorised Pearson correlation
# ---------------------------------------------------------------------------
def batch_pearsonr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson r between each (video, feature) 8-step trajectory and y.

    Args:
        X: [N, T, D]  feature trajectories
        y: [T]        reference speed profile
    Returns:
        corr: [N, D]  correlation coefficient; 0 where feature is constant
    """
    y_c = y - y.mean()                                  # [T]
    X_c = X - X.mean(axis=1, keepdims=True)             # [N, T, D]

    num   = (X_c * y_c[np.newaxis, :, np.newaxis]).sum(axis=1)   # [N, D]
    denom_X = np.sqrt((X_c ** 2).sum(axis=1))                    # [N, D]
    denom_y = float(np.sqrt((y_c ** 2).sum()))                    # scalar

    denom = denom_X * denom_y                                     # [N, D]

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / denom
    corr[~np.isfinite(corr)] = 0.0   # constant feature �?r = 0

    return corr   # [N, D]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",   required=True,
                   help="Path to the fastslow/ output directory")
    p.add_argument("--sae_path",      required=True)
    p.add_argument("--sae_type",      required=True,
                   choices=["batch_top_k", "standard"])
    p.add_argument("--output_path",   required=True)
    p.add_argument("--top_k",         default=50, type=int,
                   help="Number of top speed-correlated features to record")
    p.add_argument("--model_name",    default="MCG-NJU/videomae-base-finetuned-ssv2")
    p.add_argument("--layer",         default=11, type=int)
    p.add_argument("--attachment_point", default="post_mlp_residual")
    p.add_argument("--batch_size",    default=4,  type=int)
    p.add_argument("--num_workers",   default=4,  type=int)
    p.add_argument("--device",        default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    speed_profile = compute_tubelet_speed_profile()
    print(f"Tubelet speed profile (m/s): {speed_profile.round(3).tolist()}")

    # ── VideoMAE ──────────────────────────────────────────────────────────
    print(f"Loading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)
    model.attach(args.attachment_point, args.layer, sae=None)
    hook_key = f"{args.attachment_point}_{args.layer}"

    # ── SAE ───────────────────────────────────────────────────────────────
    print(f"Loading SAE ({args.sae_type}): {args.sae_path}")
    sae = load_sae(args.sae_type, args.sae_path, device)

    # ── Dataset ───────────────────────────────────────────────────────────
    ds = FastSlowDataset(args.dataset_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=make_collate(model.processor))

    all_features  = []
    all_labels    = []
    all_video_ids = []

    # ── Steps 1-3: extract temporal features ──────────────────────────────
    print("\nStep 1-3: VideoMAE + SAE encoding with spatial max-pool ...")
    for inputs, labels, video_ids in tqdm(dl):
        # Step 1: VideoMAE forward pass
        model.encode(inputs)
        acts = model.register[hook_key][0]   # [B, 1568, 768]

        # Step 3a: reshape to [B, 8 time, 196 spatial, 768]
        B = acts.shape[0]
        acts = acts.view(B, N_TEMPORAL, N_SPATIAL, acts.shape[-1])

        # Step 3b: max pool over 196 spatial tokens �?[B, 8, 768]
        acts = acts.max(dim=2).values.to(device)

        # Step 2: SAE encode each (video, time-step) independently
        with torch.no_grad():
            acts_flat  = acts.reshape(B * N_TEMPORAL, -1)      # [B*8, 768]
            feats_flat = sae.encode(acts_flat)                  # [B*8, dict_size]
            feats      = feats_flat.reshape(B, N_TEMPORAL, -1)  # [B, 8, dict_size]

        all_features.append(feats.cpu().numpy())
        all_labels.extend(labels)
        all_video_ids.extend(video_ids)

    features = np.concatenate(all_features, axis=0)  # [N, 8, dict_size]
    N, T, D = features.shape
    print(f"Feature tensor: {features.shape}  "
          f"({N} videos × {T} time steps × {D} features)")

    # ── Step 4: speed correlation ──────────────────────────────────────────
    print("\nStep 4: Computing per-video speed correlations ...")
    corr_per_video = batch_pearsonr(features, speed_profile)  # [N, D]

    # ── Step 5: consistency check ──────────────────────────────────────────
    print("Step 5: Ranking features by consistency ...")
    corr_mean     = corr_per_video.mean(axis=0)          # [D]
    corr_std      = corr_per_video.std(axis=0)           # [D]
    corr_frac_pos = (corr_per_video > 0).mean(axis=0)   # [D]

    top_k = min(args.top_k, D)
    top_k_indices = np.argsort(corr_mean)[::-1][:top_k]

    # Print summary
    print(f"\n{'─'*60}")
    print(f"Top-{top_k} speed-correlated SAE features")
    print(f"{'rank':>4}  {'feat':>6}  {'mean_r':>7}  {'std_r':>6}  {'frac>0':>7}")
    print(f"{'─'*60}")
    for rank, fidx in enumerate(top_k_indices[:20], 1):
        print(f"{rank:4d}  {fidx:6d}  {corr_mean[fidx]:7.4f}  "
              f"{corr_std[fidx]:6.4f}  {corr_frac_pos[fidx]:7.3f}")
    if top_k > 20:
        print(f"  �?({top_k - 20} more in output pkl)")
    print(f"{'─'*60}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "features":       features,
        "labels":         all_labels,
        "video_ids":      all_video_ids,
        "speed_profile":  speed_profile,
        "corr_per_video": corr_per_video,
        "corr_mean":      corr_mean,
        "corr_std":       corr_std,
        "corr_frac_pos":  corr_frac_pos,
        "top_k_indices":  top_k_indices,
        "sae_type":       args.sae_type,
        "sae_path":       args.sae_path,
        "model_name":     args.model_name,
        "layer":          args.layer,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved �?{out_path}")


if __name__ == "__main__":
    main()
