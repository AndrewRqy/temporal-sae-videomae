"""
NFP temporal covariance test — DINOv2 SAE (negative control).

DINOv2 processes each frame independently (no temporal context).  Its SAE
features therefore cannot track any velocity-derived temporal concept.

The test computes the same within-video covariance statistic as nfp_test.py:
  C_i(V, tau) = Cov_t( psi_i(V,t), tau(V,t) )

For DINOv2, psi_i(V, t) is the SAE activation of the pooler_output for
frame 2t (first frame of each VideoMAE tubelet), processed independently.
Because DINOv2 has no temporal context, the temporal sequence of activations
carries no velocity information — so C_i -> 0 for ALL features and ALL taus.

This is the empirical negative control.  The expected result:
  - No features significant for any tau variable
  - Mean |t-stat| near zero across all features and taus

Output format is identical to nfp_test.py for direct comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder

TAU_KEYS   = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
N_TEMPORAL = 8
N_FRAMES   = 16


# ---------------------------------------------------------------------------
# Dataset — one frame per temporal step (frame 2t)
# ---------------------------------------------------------------------------

class NFPDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.video_dirs = sorted(root_dir.glob("v*"))
        if not self.video_dirs:
            raise FileNotFoundError(f"No video dirs found in {root_dir}")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        # Load only the representative frame for each temporal step
        rep_frames = [
            Image.open(vdir / f"rgba_{step * 2:05d}.png").convert("RGB")
            for step in range(N_TEMPORAL)
        ]
        with open(vdir / "metadata.json") as f:
            meta = json.load(f)

        traj = meta["trajectory"]
        tau_steps = []
        for step in range(N_TEMPORAL):
            frame_rec = traj[step * 2]
            tau_steps.append([frame_rec["tau"][k] for k in TAU_KEYS])

        return (
            rep_frames,
            torch.tensor(tau_steps, dtype=torch.float32),  # [8, 5]
            meta["video_id"],
        )


def make_collate(processor):
    def collate(batch):
        # Each item has 8 frames; flatten to one processor call per batch
        all_frames   = []
        n_frames_lst = []
        tau_lst      = []
        video_ids    = []
        for frames, tau, vid_id in batch:
            all_frames.extend(frames)
            n_frames_lst.append(len(frames))
            tau_lst.append(tau)
            video_ids.append(vid_id)
        inputs     = processor(images=all_frames, return_tensors="pt")
        tau_tensor = torch.stack(tau_lst)   # [B, 8, 5]
        return inputs, n_frames_lst, tau_tensor, video_ids
    return collate


# ---------------------------------------------------------------------------
# Within-video covariance for all tau variables
# ---------------------------------------------------------------------------

def within_video_covariance_all(feats: torch.Tensor,
                                 tau:   torch.Tensor) -> torch.Tensor:
    """
    feats : [B, T, D]
    tau   : [B, T, K]
    Returns C : [B, D, K]
    """
    psi_c = feats - feats.mean(dim=1, keepdim=True)
    tau_c = tau   - tau.mean(dim=1,   keepdim=True)
    return torch.einsum('btd,btk->bdk', psi_c, tau_c) / feats.shape[1]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(t_stat: np.ndarray, p_val: np.ndarray,
                 C_mean: np.ndarray, alpha: float = 0.05):
    """
    t_stat, p_val, C_mean : [D, K]
    """
    D, K = t_stat.shape
    bonf = alpha / D

    print(f"\n{'='*65}")
    print(f"NFP Test — DINOv2 SAE (negative control)  (Bonferroni p < {bonf:.2e})")
    print(f"{'='*65}")
    print(f"Expected: 0% significant features for all tau variables")
    print(f"{'Tau':<12} {'Sig+':>6} {'Sig-':>6} {'Total%':>8} {'Mean|t|':>9}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*9}")

    sig_any = np.zeros(D, dtype=bool)
    for k, name in enumerate(TAU_KEYS):
        t_k = t_stat[:, k]
        p_k = p_val[:, k]
        sp  = ((p_k < bonf) & (t_k > 0)).sum()
        sn  = ((p_k < bonf) & (t_k < 0)).sum()
        sig = (p_k < bonf).sum()
        sig_any |= (p_k < bonf)
        print(f"{name:<12} {sp:>6} {sn:>6} {100*sig/D:>7.2f}% {np.abs(t_k).mean():>9.4f}")

    print(f"\nFeatures significant for at least one tau : {sig_any.sum()} ({100*sig_any.mean():.2f}%)")
    print(f"Features non-significant for all taus     : {(~sig_any).sum()} ({100*(~sig_any).mean():.2f}%)")
    print(f"Overall mean |C_i|                        : {np.abs(C_mean).mean():.6f}")
    print(f"\n(Compare with VideoMAE SAE results — DINOv2 should show near-zero")
    print(f" significant counts, confirming the NFP test is not falsely flagging")
    print(f" temporally-blind features as temporal.)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",  required=True)
    p.add_argument("--sae_path",     required=True)
    p.add_argument("--output_path",  required=True)
    p.add_argument("--model_name",   default="dinov2-base")
    p.add_argument("--batch_size",   default=8,  type=int)
    p.add_argument("--num_workers",  default=8,  type=int)
    p.add_argument("--alpha",        default=0.05, type=float)
    p.add_argument("--device",       default="cuda:0")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading DINOv2: facebook/{args.model_name}")
    processor = AutoImageProcessor.from_pretrained(f"facebook/{args.model_name}")
    model     = Dinov2Model.from_pretrained(f"facebook/{args.model_name}").to(device)
    model.eval()

    print(f"Loading SAE: {args.sae_path}")
    sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    sae.eval()

    ds = NFPDataset(Path(args.dataset_dir))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=make_collate(processor))

    print(f"Dataset: {len(ds)} videos | Testing all {len(TAU_KEYS)} tau variables")
    print("(Each temporal step uses one independently-processed frame — no temporal context)")

    all_C   = []
    all_tau = []
    all_ids = []

    for inputs, n_frames_lst, tau, video_ids in tqdm(dl, desc="Extracting"):
        with torch.no_grad():
            embeds  = model(**{k: v.to(device) for k, v in inputs.items()}).pooler_output
            sae_out = sae.encode(embeds)   # [B*8, dict_size]

        # Re-group into [B, 8, dict_size]
        B = len(n_frames_lst)
        assert all(n == N_TEMPORAL for n in n_frames_lst)
        feats = sae_out.reshape(B, N_TEMPORAL, -1).cpu()   # [B, 8, dict_size]

        C = within_video_covariance_all(feats, tau)        # [B, D, 5]

        all_C.append(C)
        all_tau.append(tau)
        all_ids.extend(video_ids)

    C_all   = torch.cat(all_C,  dim=0)   # [N, D, 5]
    tau_all = torch.cat(all_tau,dim=0)   # [N, 8, 5]

    print(f"C tensor: {C_all.shape}  Running t-tests...")
    C_np   = C_all.numpy()
    t_stat = np.zeros((C_np.shape[1], len(TAU_KEYS)), dtype=np.float32)
    p_val  = np.ones_like(t_stat)
    for k in range(len(TAU_KEYS)):
        t_stat[:, k], p_val[:, k] = stats.ttest_1samp(C_np[:, :, k], 0.0)

    C_mean = C_np.mean(axis=0)
    print_report(t_stat, p_val, C_mean, alpha=args.alpha)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "C":        C_all,
        "tau":      tau_all,
        "t_stat":   torch.from_numpy(t_stat),
        "p_val":    torch.from_numpy(p_val),
        "C_mean":   torch.from_numpy(C_mean),
        "video_ids":all_ids,
        "tau_keys": TAU_KEYS,
        "model":    f"dinov2_{args.model_name}",
    }, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
