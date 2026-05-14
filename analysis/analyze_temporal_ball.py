"""
Temporal SAE feature analysis on ablation subsets of the ball dataset.

Instead of max-pooling all 1568 tokens, we preserve the 8 temporal positions:
  16 PNG frames â†?VideoMAE layer 11 â†?[1568, 768]
               â†?reshape [8 time, 196 spatial, 768]
               â†?mean pool spatial â†?[8, 768]
               â†?SAE encode each step â†?[8, dict_size]

Ablation mode: vary exactly one variable, fix the rest. This gives small sets
(7-8 videos) where we can directly compare temporal activation patterns.

Output .pkl per ablation:
  {
    "features":    np.ndarray [N, 8, dict_size],  # temporal SAE activations
    "labels":      list[dict],                    # ground-truth label per video
    "video_ids":   list[str],
    "ablation":    str,                           # "direction" | "speed" | "position"
    "dataset_type": str,
    "sae_type":    str,
  }

Usage:
  python analyze_ball_temporal.py \\
    --dataset_dir  /path/to/ball_dataset/velocity \\
    --sae_path     /path/to/ae.pt \\
    --sae_type     batch_top_k \\
    --ablation     direction \\
    --fix_param_idx 3 \\
    --fix_pos_idx   4 \\
    --output_path  ablation_velocity_direction_btk.pkl
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
N_TEMPORAL  = 8    # 16 frames / 2 frames per tubelet
N_SPATIAL   = 196  # 14Ã—14 patches per frame
N_TOKENS    = N_TEMPORAL * N_SPATIAL  # 1568


# ---------------------------------------------------------------------------
# Ablation video selector
# ---------------------------------------------------------------------------
def select_ablation_videos(dataset_dir: Path, dataset_type: str,
                           ablation: str, fix_dir: int, fix_param: int, fix_pos: int):
    """
    Return the list of video dirs for the requested ablation.
    One variable varies freely; the other two are fixed.
    """
    dirs = []

    if ablation == "direction":
        # vary direction 0-7, fix param and pos
        for d in range(8):
            tag = f"dir{d:02d}_param{fix_param:02d}_pos{fix_pos:02d}"
            p = dataset_dir / tag
            if p.exists():
                dirs.append(p)

    elif ablation in ("speed", "acceleration"):
        # vary param index, fix direction and pos
        n_params = 7 if dataset_type == "velocity" else 5
        for p_idx in range(n_params):
            tag = f"dir{fix_dir:02d}_param{p_idx:02d}_pos{fix_pos:02d}"
            p = dataset_dir / tag
            if p.exists():
                dirs.append(p)

    elif ablation == "position":
        # vary position 0-6, fix direction and param
        for s in range(7):
            tag = f"dir{fix_dir:02d}_param{fix_param:02d}_pos{s:02d}"
            p = dataset_dir / tag
            if p.exists():
                dirs.append(p)

    else:
        raise ValueError(f"Unknown ablation '{ablation}'. "
                         "Use: direction | speed | acceleration | position")

    if not dirs:
        raise FileNotFoundError(
            f"No videos found for ablation={ablation}, "
            f"fix_dir={fix_dir}, fix_param={fix_param}, fix_pos={fix_pos}")

    print(f"Ablation '{ablation}': {len(dirs)} videos selected")
    return sorted(dirs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AblationDataset(Dataset):
    def __init__(self, video_dirs, num_frames=16):
        self.video_dirs = video_dirs
        self.num_frames = num_frames

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
def load_sae(sae_type, sae_path, device):
    if sae_type == "batch_top_k":
        sae = BatchTopKSAE.from_pretrained(sae_path, device=device)
    elif sae_type == "standard":
        sae = AutoEncoder.from_pretrained(sae_path, device=device)
    else:
        raise ValueError(f"Unknown sae_type '{sae_type}'")
    sae.eval()
    return sae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",   required=True)
    p.add_argument("--sae_path",      required=True)
    p.add_argument("--sae_type",      required=True, choices=["batch_top_k", "standard"])
    p.add_argument("--output_path",   required=True)
    p.add_argument("--ablation",      required=True,
                   choices=["direction", "speed", "acceleration", "position"])
    p.add_argument("--fix_dir_idx",   default=0,  type=int, help="Fixed direction index (0-7)")
    p.add_argument("--fix_param_idx", default=3,  type=int, help="Fixed speed/accel index")
    p.add_argument("--fix_pos_idx",   default=4,  type=int, help="Fixed start-position index (4=center)")
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
    dataset_dir = Path(args.dataset_dir)

    # Infer dataset type from first video's metadata
    sample_meta = sorted(dataset_dir.glob("*/metadata.json"))[0]
    with open(sample_meta) as f:
        dataset_type = json.load(f)["label"]["dataset_type"]

    # Select ablation videos
    video_dirs = select_ablation_videos(
        dataset_dir, dataset_type,
        ablation=args.ablation,
        fix_dir=args.fix_dir_idx,
        fix_param=args.fix_param_idx,
        fix_pos=args.fix_pos_idx,
    )

    # VideoMAE
    print(f"Loading VideoMAE: {args.model_name}")
    model = VideoMAE(args.model_name, device)
    model.attach(args.attachment_point, args.layer, sae=None)
    hook_key = f"{args.attachment_point}_{args.layer}"

    # SAE
    print(f"Loading SAE ({args.sae_type}): {args.sae_path}")
    sae = load_sae(args.sae_type, args.sae_path, device)

    # DataLoader
    ds = AblationDataset(video_dirs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=make_collate(model.processor))

    all_features = []
    all_labels   = []
    all_video_ids = []

    print("Extracting temporal features...")
    for inputs, labels, video_ids in tqdm(dl):
        model.encode(inputs)

        # acts: [B, 1568, 768]
        acts = model.register[hook_key][0]

        # Reshape to [B, 8 time, 196 spatial, 768]
        B = acts.shape[0]
        acts = acts.view(B, N_TEMPORAL, N_SPATIAL, acts.shape[-1])

        # Max pool over spatial â†?[B, 8, 768]
        # Max preserves the strongest signal (sparse object on plain background).
        acts = acts.max(dim=2).values.to(device)

        # SAE encode each time step â†?[B, 8, dict_size]
        with torch.no_grad():
            B, T, D = acts.shape
            acts_flat = acts.reshape(B * T, D)           # [B*8, 768]
            feats_flat = sae.encode(acts_flat)            # [B*8, dict_size]
            feats = feats_flat.reshape(B, T, -1)         # [B, 8, dict_size]

        all_features.append(feats.cpu().numpy())
        all_labels.extend(labels)
        all_video_ids.extend(video_ids)

    features_array = np.concatenate(all_features, axis=0)  # [N, 8, dict_size]
    print(f"Done. Feature tensor: {features_array.shape}  "
          f"({features_array.shape[0]} videos Ã— {N_TEMPORAL} time steps Ã— {features_array.shape[2]} features)")

    out = {
        "features":     features_array,
        "labels":       all_labels,
        "video_ids":    all_video_ids,
        "ablation":     args.ablation,
        "dataset_type": dataset_type,
        "sae_type":     args.sae_type,
        "sae_path":     args.sae_path,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved â†?{out_path}")


if __name__ == "__main__":
    main()
