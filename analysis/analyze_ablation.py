"""
Temporal SAE feature analysis for ablation datasets (static / back-and-forth).

Dataset types:
  static    �?ball is motionless throughout. Speed = 0 everywhere.
              Reference: time index [0,1,...,7] (tests pure temporal-progress encoding).
              Key question: does feat 2130 ramp toward t=7 even with no motion?

  backforth �?ball moves forward frames 0-7, reverses frames 8-15.
              Three reference signals are correlated simultaneously:
                �?speed_mag   : [3.0]*8 (flat)         �?instantaneous speed hypothesis
                �?path_length : [0.25,0.5,...,2.0]      �?path-length hypothesis
                �?net_disp    : [0.25,...,1.0,...,0.0]  �?net-displacement hypothesis

Pipeline (same as other analysis scripts):
  1. VideoMAE �?[1568, 768]
  2. Reshape �?[8 time, 196 spatial, 768]
  3. Max pool spatial �?[8, 768]
  4. SAE encode per time step �?[8, dict_size]
  5. Correlate 8-step trajectory with each reference signal

Output .pkl:
  {
    "features":        [N, 8, dict_size],
    "labels":          list[dict],
    "video_ids":       list[str],
    "dataset_type":    str,
    "sae_type":        str,
    # For static:
    "corr_time_mean":  [dict_size],   corr vs linear time index
    "corr_time_std":   [dict_size],
    # For backforth:
    "corr_speed_mean":  [dict_size],  corr vs flat speed
    "corr_path_mean":   [dict_size],  corr vs path length
    "corr_disp_mean":   [dict_size],  corr vs net displacement
    "top_k_indices":   [K],           ranked by best reference correlation
  }

Usage:
  python analyze_ablation_temporal.py \\
    --dataset_type  static \\
    --dataset_dir   /path/to/static \\
    --sae_path      /path/to/ae.pt \\
    --sae_type      standard \\
    --output_path   static_std.pkl
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

N_TEMPORAL = 8
N_SPATIAL  = 196
FRAME_RATE = 24

# Back-and-forth constants (must match backforth_ball_dataset.py)
BF_FORWARD_SPEED = 3.0
BF_N_FORWARD     = 8   # tubelets 0-3 = frames 0-7

# Tubelet-level reference signals for back-and-forth
def backforth_references() -> dict:
    t = np.arange(N_TEMPORAL, dtype=np.float32)
    dt_per_tubelet = 2.0 / FRAME_RATE   # 2 frames per tubelet

    speed_mag   = np.full(N_TEMPORAL, BF_FORWARD_SPEED, dtype=np.float32)

    path_length = (t + 1) * BF_FORWARD_SPEED * dt_per_tubelet

    n_fwd_tub = BF_N_FORWARD // 2  # = 4 tubelets going forward
    disp = np.zeros(N_TEMPORAL, dtype=np.float32)
    for i in range(N_TEMPORAL):
        if i < n_fwd_tub:
            disp[i] = (i + 1) * BF_FORWARD_SPEED * dt_per_tubelet
        else:
            disp[i] = max(0.0, (2 * n_fwd_tub - 1 - i) * BF_FORWARD_SPEED * dt_per_tubelet)

    return {"speed_mag": speed_mag, "path_length": path_length, "net_disp": disp}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AblationDataset(Dataset):
    """Loads videos from static (pos*_v*/) or backforth (dir*_pos*/) layout."""

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
        raise ValueError(f"Unknown sae_type '{sae_type}'")
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Vectorised Pearson correlation
# ---------------------------------------------------------------------------
def batch_pearsonr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    X: [N, T, D]  feature trajectories
    y: [T]        reference signal
    Returns [N, D]; 0 where variance is zero.
    """
    y_c  = y - y.mean()
    X_c  = X - X.mean(axis=1, keepdims=True)
    num  = (X_c * y_c[np.newaxis, :, np.newaxis]).sum(axis=1)
    dX   = np.sqrt((X_c ** 2).sum(axis=1))
    dy   = float(np.sqrt((y_c ** 2).sum()))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / (dX * dy)
    corr[~np.isfinite(corr)] = 0.0
    return corr   # [N, D]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_top(corr_mean, top_k_indices, label: str, top_n: int = 15):
    print(f"\n{'─'*55}")
    print(f"  Top features by {label}")
    print(f"{'rank':>4}  {'feat':>6}  {'mean_r':>7}")
    print(f"{'─'*55}")
    for rank, fidx in enumerate(top_k_indices[:top_n], 1):
        print(f"{rank:4d}  {fidx:6d}  {corr_mean[fidx]:7.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_type",  required=True,
                   choices=["static", "backforth"])
    p.add_argument("--dataset_dir",   required=True)
    p.add_argument("--sae_path",      required=True)
    p.add_argument("--sae_type",      required=True,
                   choices=["batch_top_k", "standard"])
    p.add_argument("--output_path",   required=True)
    p.add_argument("--top_k",         default=100, type=int)
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

    # ── Reference signals ──────────────────────────────────────────────────
    time_ref = np.arange(N_TEMPORAL, dtype=np.float32)  # [0,1,...,7]

    if args.dataset_type == "backforth":
        bf_refs = backforth_references()
        print("Back-and-forth reference signals (per tubelet):")
        print(f"  speed_mag   : {bf_refs['speed_mag'].tolist()}")
        print(f"  path_length : {bf_refs['path_length'].round(3).tolist()}")
        print(f"  net_disp    : {bf_refs['net_disp'].round(3).tolist()}")
    else:
        print("Static dataset �?reference: linear time index [0,1,...,7]")

    # ── Models ─────────────────────────────────────────────────────────────
    print(f"\nLoading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)
    model.attach(args.attachment_point, args.layer, sae=None)
    hook_key = f"{args.attachment_point}_{args.layer}"

    print(f"Loading SAE ({args.sae_type}): {args.sae_path}")
    sae = load_sae(args.sae_type, args.sae_path, device)

    # ── Dataset ─────────────────────────────────────────────────────────────
    ds = AblationDataset(args.dataset_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=make_collate(model.processor))

    all_features  = []
    all_labels    = []
    all_video_ids = []

    # ── Steps 1-3: extract temporal features ───────────────────────────────
    print("\nSteps 1-3: VideoMAE + SAE encoding with spatial max-pool ...")
    for inputs, labels, video_ids in tqdm(dl):
        model.encode(inputs)
        acts = model.register[hook_key][0]   # [B, 1568, 768]

        B = acts.shape[0]
        acts = acts.view(B, N_TEMPORAL, N_SPATIAL, acts.shape[-1])
        acts = acts.max(dim=2).values.to(device)   # [B, 8, 768]

        with torch.no_grad():
            acts_flat  = acts.reshape(B * N_TEMPORAL, -1)
            feats_flat = sae.encode(acts_flat)
            feats      = feats_flat.reshape(B, N_TEMPORAL, -1)  # [B, 8, D]

        all_features.append(feats.cpu().numpy())
        all_labels.extend(labels)
        all_video_ids.extend(video_ids)

    features = np.concatenate(all_features, axis=0)
    N, T, D  = features.shape
    print(f"Feature tensor: {features.shape}")

    # ── Step 4-5: correlations ─────────────────────────────────────────────
    print("\nComputing correlations ...")
    out = {
        "features":     features,
        "labels":       all_labels,
        "video_ids":    all_video_ids,
        "dataset_type": args.dataset_type,
        "sae_type":     args.sae_type,
        "sae_path":     args.sae_path,
        "model_name":   args.model_name,
        "layer":        args.layer,
    }

    if args.dataset_type == "static":
        corr_time      = batch_pearsonr(features, time_ref)   # [N, D]
        corr_time_mean = corr_time.mean(axis=0)
        corr_time_std  = corr_time.std(axis=0)
        top_k_indices  = np.argsort(corr_time_mean)[::-1][:args.top_k]

        out.update({
            "time_ref":        time_ref,
            "corr_time":       corr_time,
            "corr_time_mean":  corr_time_mean,
            "corr_time_std":   corr_time_std,
            "top_k_indices":   top_k_indices,
        })
        print_top(corr_time_mean, top_k_indices, "time-index correlation")

    else:  # backforth
        for ref_name, ref_signal in bf_refs.items():
            corr = batch_pearsonr(features, ref_signal)
            out[f"corr_{ref_name}"]      = corr
            out[f"corr_{ref_name}_mean"] = corr.mean(axis=0)
            out[f"corr_{ref_name}_std"]  = corr.std(axis=0)

        # Rank by net_disp correlation (the most discriminative signal)
        top_by_disp  = np.argsort(out["corr_net_disp_mean"])[::-1][:args.top_k]
        top_by_path  = np.argsort(out["corr_path_length_mean"])[::-1][:args.top_k]
        top_by_speed = np.argsort(out["corr_speed_mag_mean"])[::-1][:args.top_k]

        out["top_k_indices"]         = top_by_disp
        out["top_k_by_path_length"]  = top_by_path
        out["top_k_by_speed_mag"]    = top_by_speed
        out["bf_refs"]               = bf_refs

        print_top(out["corr_net_disp_mean"],    top_by_disp,  "net displacement")
        print_top(out["corr_path_length_mean"], top_by_path,  "path length")
        print_top(out["corr_speed_mag_mean"],   top_by_speed, "speed magnitude")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved �?{out_path}")


if __name__ == "__main__":
    main()
