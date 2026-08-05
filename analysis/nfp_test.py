"""
NFP temporal covariance test — VideoMAE SAE.

Proves two claims simultaneously:
  (1) Neurons encoding a temporal concept tau show statistically significant
      within-video covariance with tau's sequence: mean_V C_i(V, tau) != 0
  (2) Neurons not encoding tau show covariance -> 0 (converges with N)
  (3) Selectivity: neurons significant for tau_A are NOT significant for tau_B
      when A and B are unrelated concepts (speed vs direction, etc.)

Test statistic (Section 4 of proof):
  C_i(V, tau) = Cov_t( psi_i(V,t), tau(V,t) )
              = (1/T) sum_t [ (psi_i - mean psi_i) * (tau - mean tau) ]

psi_i(V,t) = ball-TRACKING activation — feature i at the spatial token
             containing the ball at temporal step t, NOT max-pooled.
             Zero when ball is off-screen (compact support A1).

All 5 tau variables tested simultaneously. One-sample t-test with Bonferroni
correction. Report includes cross-tau selectivity for significant features.

NOTE: Pearson correlation is deliberately NOT used — the appendix proves it
produces false positives for static features (denominator Var_t(psi) depends
on x0 and breaks the required linearity).
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from dictionary_learning import AutoEncoder, PCADict, ICADict, IdentityDict
from models.videomae import VideoMAE

TAU_KEYS   = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
N_TEMPORAL = 8
N_SPATIAL  = 196
N_FRAMES   = 16


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NFPDataset(Dataset):
    def __init__(self, root_dir: Path, tau_mode: str = "first_frame"):
        self.video_dirs = sorted(root_dir.glob("v*"))
        self.tau_mode   = tau_mode
        if not self.video_dirs:
            raise FileNotFoundError(f"No video dirs found in {root_dir}")

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        frames = [
            Image.open(vdir / f"rgba_{i:05d}.png").convert("RGB")
            for i in range(N_FRAMES)
        ]
        with open(vdir / "metadata.json") as f:
            meta = json.load(f)

        traj = meta["trajectory"]
        tau_steps        = []   # [8, 5]
        ball_token_steps = []   # [8]  -1 if off-screen

        for step in range(N_TEMPORAL):
            r0 = traj[step * 2]       # first frame of tubelet
            r1 = traj[step * 2 + 1]  # second frame of tubelet
            if self.tau_mode == "avg_frames":
                tau_steps.append([(r0["tau"][k] + r1["tau"][k]) / 2 for k in TAU_KEYS])
            else:  # first_frame (default)
                tau_steps.append([r0["tau"][k] for k in TAU_KEYS])
            ball_token_steps.append(r0["spatial_token"])  # always use first frame for position

        return (
            frames,
            torch.tensor(tau_steps,        dtype=torch.float32),  # [8, 5]
            torch.tensor(ball_token_steps, dtype=torch.long),      # [8]
            meta["video_id"],
        )


def make_collate(processor):
    def collate(batch):
        inputs      = processor(images=[b[0] for b in batch], return_tensors="pt")
        tau         = torch.stack([b[1] for b in batch])   # [B, 8, 5]
        ball_tokens = torch.stack([b[2] for b in batch])   # [B, 8]
        video_ids   = [b[3] for b in batch]
        return inputs, tau, ball_tokens, video_ids
    return collate


# ---------------------------------------------------------------------------
# Ball-tracking extraction
# ---------------------------------------------------------------------------

def extract_ball_tracking(acts_spatial: torch.Tensor,
                           ball_tokens:  torch.Tensor):
    """
    acts_spatial : [B, 8, 196, D]
    ball_tokens  : [B, 8]   spatial token index, -1 = off-screen

    Returns:
        ball_acts : [B, 8, D]  activation at ball token (0 if off-screen)
        mask      : [B, 8]     True where ball is on-screen
    """
    B, T, S, D = acts_spatial.shape
    mask        = (ball_tokens >= 0)
    safe_tokens = ball_tokens.clamp(min=0)                           # [B, 8]
    idx         = safe_tokens.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, D)  # [B, 8, 1, D]
    ball_acts   = acts_spatial.gather(2, idx).squeeze(2)            # [B, 8, D]
    ball_acts   = ball_acts * mask.unsqueeze(-1).float()            # zero off-screen
    return ball_acts, mask


# ---------------------------------------------------------------------------
# Within-video covariance for all tau variables at once
# ---------------------------------------------------------------------------

def within_video_covariance_all(feats: torch.Tensor,
                                 tau:   torch.Tensor) -> torch.Tensor:
    """
    feats : [B, T, D]   ball-tracking SAE activations
    tau   : [B, T, K]   all tau variables

    Returns C : [B, D, K]  within-video covariance per video, feature, tau.
    """
    psi_c = feats - feats.mean(dim=1, keepdim=True)      # [B, T, D]
    tau_c = tau   - tau.mean(dim=1,   keepdim=True)      # [B, T, K]

    # C[b, d, k] = (1/T) sum_t psi_c[b,t,d] * tau_c[b,t,k]
    C = torch.einsum('btd,btk->bdk', psi_c, tau_c) / feats.shape[1]  # [B, D, K]
    return C


# ---------------------------------------------------------------------------
# Report: claims (1), (2), (3)
# ---------------------------------------------------------------------------

def print_report(t_stat: np.ndarray, p_val: np.ndarray,
                 C_mean: np.ndarray, alpha: float = 0.05,
                 label: str = "VideoMAE SAE", tau_sigma: np.ndarray = None):
    """
    t_stat, p_val, C_mean : [D, K]
    tau_sigma : [K] global per-tau std, for the scale-normalized selectivity matrix.
    """
    D, K = t_stat.shape
    bonf = alpha / D

    print(f"\n{'='*65}")
    print(f"NFP Test — {label}  (Bonferroni threshold p < {bonf:.2e})")
    print(f"{'='*65}")
    print(f"{'Tau':<12} {'Sig+':>6} {'Sig-':>6} {'Total%':>8} {'Mean|t|':>9}")
    print(f"{'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*9}")

    sig_any = np.zeros(D, dtype=bool)
    for k, name in enumerate(TAU_KEYS):
        t_k  = t_stat[:, k]
        p_k  = p_val[:, k]
        sp   = ((p_k < bonf) & (t_k > 0)).sum()
        sn   = ((p_k < bonf) & (t_k < 0)).sum()
        sig  = (p_k < bonf).sum()
        sig_any |= (p_k < bonf)
        print(f"{name:<12} {sp:>6} {sn:>6} {100*sig/D:>7.2f}% {np.abs(t_k).mean():>9.4f}")

    print(f"\nFeatures significant for at least one tau : {sig_any.sum()} ({100*sig_any.mean():.2f}%)")
    print(f"Features non-significant for all taus     : {(~sig_any).sum()} ({100*(~sig_any).mean():.2f}%)")

    # Claim (2): non-significant features converge to 0
    non_sig_C = C_mean[~sig_any]
    print(f"\nClaim (2) — non-significant features:")
    print(f"  Mean |C_mean| across all taus : {np.abs(non_sig_C).mean():.6f}")

    # Claim (3): selectivity matrix — for features significant in tau A, how strongly
    # do they respond to tau B? Reported as mean |C_mean| with tau Z-SCORED globally
    # (equivalently: raw-mean-covariance column divided by the global sigma of that tau),
    # so columns are effect sizes on one common scale. NOT mean |t|: the t-stat measures
    # consistency-across-videos, not response strength, and would let a tiny-but-reliable
    # covariance dominate the matrix. Flagging (rows) stays t-based — that is the NFP
    # guarantee; the matrix itself is a post-hoc effect-size diagnostic.
    C_sel = C_mean / tau_sigma[None, :] if tau_sigma is not None else C_mean
    unit = "z-scored tau" if tau_sigma is not None else "RAW tau — pass tau_sigma!"
    print(f"\nClaim (3) — selectivity (mean |C_mean| on {unit}, sig-in-row across columns):")
    header = f"{'':12}" + "".join(f"{n:>11}" for n in TAU_KEYS)
    print(header)
    for k_row, name_row in enumerate(TAU_KEYS):
        sig_mask = (p_val[:, k_row] < bonf)
        if sig_mask.sum() == 0:
            row = f"{name_row:<12}" + "".join(f"{'(none)':>11}" for _ in TAU_KEYS)
        else:
            row = f"{name_row:<12}"
            for k_col, name_col in enumerate(TAU_KEYS):
                mc = np.abs(C_sel[sig_mask, k_col]).mean()
                row += f"{mc:>10.4f}{' <--' if k_col == k_row else '    '}"
        print(row)

    # Top features per tau
    for k, name in enumerate(TAU_KEYS):
        sig_mask = (p_val[:, k] < bonf) & (t_stat[:, k] > 0)
        if sig_mask.sum() == 0:
            print(f"\n{name}: no significant positive features")
            continue
        top_idx = np.argsort(-t_stat[:, k])[:5]
        print(f"\nTop 5 features for {name} (by t-stat):")
        print(f"  {'feat':>6} {'t-stat':>8} {'p':>10} " +
              "  ".join(f"{n:>10}" for n in TAU_KEYS))
        for d in top_idx:
            t_row = "  ".join(f"{t_stat[d,kk]:>+10.3f}" for kk in range(K))
            print(f"  {d:>6} {t_stat[d,k]:>8.3f} {p_val[d,k]:>10.2e}  {t_row}"
                  f"  {'*SIG*' if p_val[d,k] < bonf else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",      required=True)
    p.add_argument("--sae_path",         default=None,
                   help="Path to the dictionary checkpoint. Not needed for --sae_model identity.")
    p.add_argument("--sae_model",        default="standard",
                   choices=["standard", "pca", "ica", "identity"],
                   help="standard=ReLU SAE; pca/ica=linear decomposition; "
                        "identity=raw layer dimensions (no transform).")
    p.add_argument("--decomp_mode",      default="sign_split",
                   choices=["sign_split", "abs", "signed"],
                   help="For pca/ica: how signed components map to features "
                        "(sign_split=ReLU(+c)/ReLU(-c), matching the ReLU SAE).")
    p.add_argument("--output_path",      required=True)
    p.add_argument("--model_name",       default="MCG-NJU/videomae-base-finetuned-ssv2")
    p.add_argument("--layer",            default=11, type=int)
    p.add_argument("--attachment_point", default="post_mlp_residual")
    p.add_argument("--batch_size",       default=4,  type=int)
    p.add_argument("--num_workers",      default=4,  type=int)
    p.add_argument("--alpha",            default=0.05, type=float)
    p.add_argument("--label",            default="VideoMAE SAE")
    p.add_argument("--device",           default="cuda:0")
    p.add_argument("--tau_mode",         default="first_frame",
                   choices=["first_frame", "avg_frames"],
                   help="first_frame: use tau at first frame of each tubelet; "
                        "avg_frames: average tau across both frames of each tubelet")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading VideoMAE: {args.model_name}")
    model    = VideoMAE(args.model_name, device)
    hook_key = f"{args.attachment_point}_{args.layer}"
    model.attach(args.attachment_point, args.layer, sae=None)

    print(f"Loading dictionary ({args.sae_model}): {args.sae_path}")
    if args.sae_model == "standard":
        sae = AutoEncoder.from_pretrained(args.sae_path, device=device)
    elif args.sae_model == "pca":
        sae = PCADict.from_pretrained(args.sae_path, device=device, mode=args.decomp_mode)
    elif args.sae_model == "ica":
        sae = ICADict.from_pretrained(args.sae_path, device=device, mode=args.decomp_mode)
    elif args.sae_model == "identity":
        # Raw layer dimensions: encode is the identity, so feats == ball-tracking
        # activations (768 raw VideoMAE dims). The NFP analog of the raw MS baseline.
        sae = IdentityDict.from_pretrained(None)
    else:
        raise ValueError(f"Unknown sae_model: {args.sae_model}")
    sae.eval()

    print(f"Tau mode: {args.tau_mode}")
    ds = NFPDataset(Path(args.dataset_dir), tau_mode=args.tau_mode)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=make_collate(model.processor))

    print(f"Dataset: {len(ds)} videos | Testing all {len(TAU_KEYS)} tau variables")

    all_C    = []
    all_tau  = []
    all_mask = []
    all_ids  = []

    for inputs, tau, ball_tokens, video_ids in tqdm(dl, desc="Extracting"):
        model.encode(inputs)
        acts         = model.register[hook_key][0]            # [B, 1568, 768]
        B            = acts.shape[0]
        acts_spatial = acts.view(B, N_TEMPORAL, N_SPATIAL, -1)

        ball_acts, mask = extract_ball_tracking(
            acts_spatial.to(device), ball_tokens.to(device)
        )  # [B, 8, 768], [B, 8]

        with torch.no_grad():
            B2, T, Dh   = ball_acts.shape
            feats_flat  = sae.encode(ball_acts.reshape(B2 * T, Dh))
            feats       = feats_flat.reshape(B2, T, -1).cpu()

        feats = feats * mask.cpu().unsqueeze(-1).float()
        C     = within_video_covariance_all(feats, tau)   # [B, D, 5]

        all_C.append(C)
        all_tau.append(tau)
        all_mask.append(mask)
        all_ids.extend(video_ids)

    C_all    = torch.cat(all_C,   dim=0)   # [N, D, 5]
    tau_all  = torch.cat(all_tau, dim=0)   # [N, 8, 5]
    mask_all = torch.cat(all_mask,dim=0)   # [N, 8]

    print(f"C tensor: {C_all.shape}  Running t-tests...")
    C_np = C_all.numpy()                              # [N, D, 5]
    t_stat = np.zeros((C_np.shape[1], len(TAU_KEYS)), dtype=np.float32)
    p_val  = np.ones_like(t_stat)
    for k in range(len(TAU_KEYS)):
        t_stat[:, k], p_val[:, k] = stats.ttest_1samp(C_np[:, :, k], 0.0)

    C_mean = C_np.mean(axis=0)   # [D, 5]
    # global per-tau std over all videos x steps: converts raw-covariance columns to
    # z-scored-tau units for the selectivity matrix (Cov(psi, tau/sigma) = Cov/sigma)
    tau_sigma = tau_all.reshape(-1, len(TAU_KEYS)).numpy().std(axis=0)
    print_report(t_stat, p_val, C_mean, alpha=args.alpha, label=args.label,
                 tau_sigma=tau_sigma)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "C":         C_all,
        "tau":       tau_all,
        "ball_mask": mask_all,
        "t_stat":    torch.from_numpy(t_stat),
        "p_val":     torch.from_numpy(p_val),
        "C_mean":    torch.from_numpy(C_mean),
        "video_ids": all_ids,
        "tau_keys":  TAU_KEYS,
    }, out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
