"""
Generate synthetic video representations for the oracle positive/negative control.

For each video V at temporal step t:

    h(V,t) = tau_norm(V,t) @ W_tau          [TEMPORALLY VARYING — 5 directions]
           + static_code(V) @ W_static       [STATIC — N_STATIC directions, random per video]
           + noise

Where:
  tau_norm(V,t)   - Z-scored tau variables at step t (changes as ball moves)
  static_code(V)  - random vector drawn once per video (never changes within V)
  W_tau    [5,768]         - orthonormal temporal directions
  W_static [N_STATIC,768] - orthonormal static directions (orthogonal to W_tau)
  noise           - small i.i.d. Gaussian

N_STATIC >> 5 so the SAE has much more static content to encode than temporal,
mimicking the real VideoMAE setting where scene/object content dominates and
motion signal is a small fraction. This forces the SAE to spread its features
across many static directions, leaving fewer dedicated to the temporal component —
pushing the significant-feature percentage closer to VideoMAE's 1.22%.

NFP should still find the temporal features among the crowd, and correctly
ignore all N_STATIC static directions.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

TAU_KEYS   = ["speed", "vel_x", "vel_y", "accel_mag", "direction"]
N_TEMPORAL = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nfp_dir",      required=True,  help="NFP dataset root (v0000/, v0001/, ...)")
    p.add_argument("--output_dir",   required=True,  help="Where to save activations and matrices")
    p.add_argument("--dim",          default=768,    type=int, help="Representation dimension")
    p.add_argument("--n_static",     default=100,    type=int,
                   help="Number of static directions (>> 5 to mimic VideoMAE)")
    p.add_argument("--noise_scale",  default=0.05,   type=float)
    p.add_argument("--seed",         default=42,     type=int)
    p.add_argument("--val_frac",     default=0.2,    type=float)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    nfp_dir    = Path(args.nfp_dir)
    video_dirs = sorted(nfp_dir.glob("v*"))
    N          = len(video_dirs)
    D          = args.dim
    N_STATIC   = args.n_static
    print(f"Found {N} videos | dim={D} | n_static={N_STATIC}")

    # --- Load all tau values [N, 8, 5] from metadata ---
    all_tau = []
    for vdir in tqdm(video_dirs, desc="Loading metadata"):
        with open(vdir / "metadata.json") as f:
            meta = json.load(f)
        traj = meta["trajectory"]
        steps = []
        for step in range(N_TEMPORAL):
            rec = traj[step * 2]
            steps.append([rec["tau"][k] for k in TAU_KEYS])
        all_tau.append(steps)

    tau = torch.tensor(all_tau, dtype=torch.float32)   # [N, 8, 5]

    # --- Global Z-score normalization (using all N*8 frames) ---
    tau_flat        = tau.reshape(-1, 5)
    tau_mean_global = tau_flat.mean(0)                  # [5]
    tau_std_global  = tau_flat.std(0).clamp(min=1e-6)   # [5]
    tau_norm        = (tau - tau_mean_global) / tau_std_global  # [N, 8, 5]

    # --- Random orthonormal projection matrices ---
    # QR on a tall random matrix gives 5 + N_STATIC orthonormal columns in R^D
    Q, _ = torch.linalg.qr(torch.randn(D, 5 + N_STATIC))  # [D, 5+N_STATIC]
    W_tau    = Q[:, :5].T          # [5,        D]  temporal subspace
    W_static = Q[:, 5:].T         # [N_STATIC, D]  static subspace (orthog. to W_tau)

    # --- Temporal component: tau_norm projected through W_tau ---
    # Shape [N, 8, D] — different at each step because tau changes
    h_temporal = tau_norm @ W_tau       # [N, 8, D]

    # --- Static component: one random code vector per video, never changes within V ---
    # Each video gets a random N_STATIC-dim code, projected into representation space.
    # Constant across time steps → within-video covariance with tau = 0 by definition.
    static_codes = torch.randn(N, N_STATIC)              # [N, N_STATIC]
    h_static_per_video = static_codes @ W_static          # [N, D]
    h_static = h_static_per_video.unsqueeze(1).expand(N, N_TEMPORAL, D)  # [N, 8, D]

    noise = torch.randn(N, N_TEMPORAL, D) * args.noise_scale
    h     = h_temporal + h_static + noise                 # [N, 8, D]

    print(f"\nSignal diagnostics:")
    print(f"  Temporal component std : {h_temporal.std():.4f}  ({5} directions)")
    print(f"  Static component std   : {h_static.std():.4f}  ({N_STATIC} directions)")
    print(f"  Noise std              : {noise.std():.4f}")
    print(f"  Total h std            : {h.std():.4f}")
    print(f"  Static/temporal ratio  : {h_static.std()/h_temporal.std():.2f}x")

    # --- Save projection matrices + normalization for NFP test ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "W_tau":           W_tau,
        "W_static":        W_static,
        "tau_mean_global": tau_mean_global,
        "tau_std_global":  tau_std_global,
        "tau_keys":        TAU_KEYS,
        "dim":             D,
        "n_static":        N_STATIC,
        "seed":            args.seed,
        "noise_scale":     args.noise_scale,
    }, out_dir / "matrices.pt")

    # Save the full [N, 8, D] structure for NFP test (video-level grouping needed)
    # Filename starts with 'all' so ActivationsDataset ignores it
    torch.save({
        "h":         h,           # [N, 8, D]
        "tau":       tau,         # [N, 8, 5]  original (unnormalized)
        "tau_norm":  tau_norm,    # [N, 8, 5]  Z-scored
        "video_ids": [vd.name for vd in video_dirs],
    }, out_dir / "all_videos.pt")

    # --- Chunked flat activations for SAE training ---
    n_val   = int(N * args.val_frac)
    n_train = N - n_val
    perm    = torch.randperm(N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    h_train = h[train_idx].reshape(-1, D)   # [n_train*8, D]
    h_val   = h[val_idx].reshape(-1, D)     # [n_val*8,   D]

    def save_chunks(data, split_dir, chunk_size=4096):
        split_dir.mkdir(parents=True, exist_ok=True)
        n = data.shape[0]
        for i, start in enumerate(range(0, n, chunk_size)):
            torch.save(data[start:start + chunk_size],
                       split_dir / f"activations_part{i}.pt")
        print(f"  Saved {i+1} chunk(s) -> {split_dir}")

    print(f"\nSaving SAE training chunks...")
    save_chunks(h_train, out_dir / "train")
    save_chunks(h_val,   out_dir / "val")

    print(f"\nDone.")
    print(f"  Train: {n_train} videos  ({h_train.shape[0]} time-steps)")
    print(f"  Val:   {n_val}   videos  ({h_val.shape[0]} time-steps)")
    print(f"  W_tau and W_static saved to: {out_dir / 'matrices.pt'}")
    print(f"  Full video structure saved to: {out_dir / 'all_videos.pt'}")


if __name__ == "__main__":
    main()
