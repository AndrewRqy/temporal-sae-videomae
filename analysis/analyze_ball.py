"""
Feed ball dataset videos through VideoMAE + SAE and record per-video feature activations.

Pipeline per video:
  16 PNG frames â†?VideoMAE layer 11 (post_mlp_residual) [1568, 768]
               â†?max pool over tokens â†?[768]
               â†?SAE encoder â†?[6144] sparse features

Output: a .pkl file containing:
  {
    "features":     np.ndarray [N, dict_size],   # SAE activations
    "video_ids":    list[str],                   # e.g. "dir00_param03_pos04"
    "labels":       list[dict],                  # full metadata label per video
    "dataset_type": str,                         # "velocity" or "acceleration"
    "sae_type":     str,
    "sae_path":     str,
  }

Usage (cluster):
  python analyze_ball_dataset.py \\
    --dataset_dir /path/to/ball_dataset/velocity \\
    --sae_path    /path/to/train_batch_top_k_64_x8/trainer_0/ae.pt \\
    --sae_type    batch_top_k \\
    --output_path ball_velocity_btk.pkl

  python analyze_ball_dataset.py \\
    --dataset_dir /path/to/ball_dataset/acceleration \\
    --sae_path    /path/to/train_standard_8_x8/trainer_0/ae.pt \\
    --sae_type    standard \\
    --output_path ball_accel_standard.pkl
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BallVideoDataset(Dataset):
    """Loads pre-rendered ball videos (16 rgba PNG frames + metadata.json)."""

    def __init__(self, dataset_dir: str, num_frames: int = 16):
        self.num_frames = num_frames
        root = Path(dataset_dir)
        self.videos = sorted([
            d for d in root.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])
        if not self.videos:
            raise FileNotFoundError(f"No video folders found in {dataset_dir}")
        print(f"Found {len(self.videos)} videos in {dataset_dir}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vdir = self.videos[idx]
        frames = []
        for i in range(self.num_frames):
            img_path = vdir / f"rgba_{i:05d}.png"
            img = Image.open(img_path).convert("RGB")
            frames.append(img)

        with open(vdir / "metadata.json") as f:
            meta = json.load(f)

        return frames, meta["label"], vdir.name


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
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",  required=True,
                   help="Path to velocity/ or acceleration/ folder")
    p.add_argument("--sae_path",     required=True,
                   help="Path to ae.pt SAE checkpoint")
    p.add_argument("--sae_type",     required=True,
                   choices=["batch_top_k", "standard"])
    p.add_argument("--output_path",  required=True,
                   help="Output .pkl file path")
    p.add_argument("--model_name",   default="MCG-NJU/videomae-base",
                   help="HuggingFace VideoMAE model name")
    p.add_argument("--layer",        default=11, type=int,
                   help="VideoMAE encoder layer to hook (0-indexed)")
    p.add_argument("--attachment_point", default="post_mlp_residual")
    p.add_argument("--batch_size",   default=8, type=int)
    p.add_argument("--num_workers",  default=4, type=int)
    p.add_argument("--device",       default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    # --- Dataset
    ds = BallVideoDataset(args.dataset_dir)
    with open(ds.videos[0] / "metadata.json") as f:
        dataset_type = json.load(f)["label"]["dataset_type"]

    # --- VideoMAE model
    print(f"Loading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)
    model.attach(args.attachment_point, args.layer, sae=None)  # raw activations
    hook_key = f"{args.attachment_point}_{args.layer}"

    # --- SAE
    print(f"Loading SAE ({args.sae_type}): {args.sae_path}")
    sae = load_sae(args.sae_type, args.sae_path, device)

    # --- DataLoader
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(model.processor),
    )

    all_features = []
    all_video_ids = []
    all_labels = []

    print("Extracting features...")
    for inputs, labels, video_ids in tqdm(dl):
        # --- VideoMAE forward pass (fills register; encode() resets it first)
        model.encode(inputs)

        # register holds one [B, 1568, 768] tensor appended by the hook
        acts = model.register[hook_key][0]  # [B, 1568, 768]

        # Max pool over 1568 spatio-temporal tokens â†?[B, 768]
        acts = acts.max(dim=1).values.to(device)

        # SAE encode â†?[B, dict_size]
        with torch.no_grad():
            features = sae.encode(acts)  # [B, dict_size]

        all_features.append(features.cpu().numpy())
        all_video_ids.extend(video_ids)
        all_labels.extend(labels)

    features_array = np.concatenate(all_features, axis=0)  # [N, dict_size]
    print(f"Done. Feature matrix: {features_array.shape}")

    # --- Save
    output = {
        "features":     features_array,
        "video_ids":    all_video_ids,
        "labels":       all_labels,
        "dataset_type": dataset_type,
        "sae_type":     args.sae_type,
        "sae_path":     args.sae_path,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
