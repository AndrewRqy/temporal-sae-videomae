"""
Visualize temporal SAE feature analysis results for the non-monotonic ball dataset.

Produces four figures:
  1. Correlation distribution  — histogram of mean_corr across all features
  2. Top-feature trajectories  — per-video 8-step activation vs speed profile
  3. Consistency scatter        — mean_corr vs std_corr (want top-right, narrow)
  4. Activation heatmap         — top features × all videos at each time step

Usage:
  python visualize_nonmono.py nonmono_std.pkl --top_k 10 --out_dir figures/
  python visualize_nonmono.py nonmono_btk.pkl --top_k 10 --out_dir figures/
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

TIMESTEP_LABELS = [f"t{i}" for i in range(8)]
SPEED_LABEL = [1.5, 1.5, 3.75, 6.0, 6.0, 3.75, 1.5, 1.5]   # m/s per tubelet


def load(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ── Figure 1: correlation distribution ──────────────────────────────────────
def plot_corr_distribution(data: dict, out_dir: Path, tag: str):
    corr_mean = data["corr_mean"]
    top_idx   = data["top_k_indices"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(corr_mean, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(corr_mean[top_idx[0]], color="crimson", lw=1.5,
               label=f"rank-1 feature (r={corr_mean[top_idx[0]]:.3f})")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Mean Pearson r (feature trajectory vs speed profile)")
    ax.set_ylabel("Number of SAE features")
    ax.set_title(f"Speed-correlation distribution  [{tag}]")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"01_corr_distribution_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 2: top-feature temporal trajectories ─────────────────────────────
def plot_top_trajectories(data: dict, out_dir: Path, tag: str, top_k: int = 9):
    features   = data["features"]          # [N, 8, dict_size]
    top_idx    = data["top_k_indices"][:top_k]
    corr_mean  = data["corr_mean"]
    corr_std   = data["corr_std"]
    speed_ref  = data["speed_profile"]     # [8]

    t = np.arange(8)
    ncols = 3
    nrows = int(np.ceil(top_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3),
                             sharex=True)
    axes = axes.flatten()

    for rank, (ax, fidx) in enumerate(zip(axes, top_idx)):
        traj = features[:, :, fidx]      # [N, 8]
        mu   = traj.mean(axis=0)
        sigma = traj.std(axis=0)

        # Per-video trajectories (light)
        for v in range(traj.shape[0]):
            ax.plot(t, traj[v], color="steelblue", alpha=0.15, lw=0.8)

        # Mean ± std band
        ax.fill_between(t, mu - sigma, mu + sigma, color="steelblue", alpha=0.3)
        ax.plot(t, mu, color="steelblue", lw=2, label="mean activation")

        # Speed reference (normalised to same scale)
        s_norm = (speed_ref - speed_ref.min()) / (speed_ref.max() - speed_ref.min())
        act_range = max(mu.max() - mu.min(), 1e-6)
        s_scaled  = s_norm * act_range + mu.min()
        ax.plot(t, s_scaled, color="crimson", lw=1.5, ls="--", label="speed (scaled)")

        ax.set_title(f"rank {rank+1}  feat {fidx}\n"
                     f"r̄={corr_mean[fidx]:.3f}  σ={corr_std[fidx]:.3f}",
                     fontsize=9)
        ax.set_xticks(t)
        ax.set_xticklabels(TIMESTEP_LABELS, fontsize=7)
        if rank == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for ax in axes[top_k:]:
        ax.set_visible(False)

    fig.suptitle(f"Top-{top_k} speed-correlated SAE features  [{tag}]\n"
                 "Blue=feature activation  Red dashed=speed profile (scaled)",
                 fontsize=10)
    fig.tight_layout()
    path = out_dir / f"02_top_trajectories_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 3: consistency scatter ───────────────────────────────────────────
def plot_consistency_scatter(data: dict, out_dir: Path, tag: str, top_k: int = 20):
    corr_mean    = data["corr_mean"]
    corr_std     = data["corr_std"]
    corr_frac    = data["corr_frac_pos"]
    top_idx      = data["top_k_indices"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(corr_mean, corr_std, c=corr_frac, cmap="RdYlGn",
                    s=6, alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="Fraction of videos with r > 0")

    # Annotate top features
    for rank, fidx in enumerate(top_idx[:top_k]):
        ax.annotate(f"{fidx}", (corr_mean[fidx], corr_std[fidx]),
                    fontsize=6, color="navy", alpha=0.8)

    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Mean Pearson r  (higher = more speed-responsive)")
    ax.set_ylabel("Std of Pearson r  (lower = more consistent)")
    ax.set_title(f"Feature consistency  [{tag}]\n"
                 "Ideal: top-right corner, green colour")
    fig.tight_layout()
    path = out_dir / f"03_consistency_scatter_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Figure 4: activation heatmap ────────────────────────────────────────────
def plot_heatmap(data: dict, out_dir: Path, tag: str, top_k: int = 20):
    features  = data["features"]          # [N, 8, dict_size]
    top_idx   = data["top_k_indices"][:top_k]
    video_ids = data["video_ids"]
    labels    = data["labels"]
    speed_ref = data["speed_profile"]

    # Sort videos by direction for visual grouping
    dir_order = np.argsort([lbl["direction_idx"] for lbl in labels])
    video_ids_sorted = [video_ids[i] for i in dir_order]

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(top_k, 1, hspace=0.05)

    for row, fidx in enumerate(top_idx):
        ax = fig.add_subplot(gs[row])
        traj = features[dir_order, :, fidx]    # [N, 8] sorted by direction
        im = ax.imshow(traj, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(f"f{fidx}", fontsize=6, rotation=0, labelpad=28, va="center")
        if row < top_k - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks(range(8))
            ax.set_xticklabels(TIMESTEP_LABELS, fontsize=7)
            ax.set_xlabel("Tubelet time step")

    fig.suptitle(f"Top-{top_k} feature activations across {len(video_ids)} videos  [{tag}]\n"
                 "Each row = one SAE feature, each column = time step, "
                 "rows grouped by direction",
                 fontsize=9)
    # Speed profile bar at top
    ax_speed = fig.add_axes([0.125, 0.92, 0.775, 0.04])
    ax_speed.bar(range(8), speed_ref, color="crimson", alpha=0.7, width=0.8)
    ax_speed.set_xlim(-0.5, 7.5)
    ax_speed.set_yticks([])
    ax_speed.set_xticks([])
    ax_speed.set_title("Speed profile (m/s)", fontsize=7, pad=2)

    path = out_dir / f"04_heatmap_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Summary table ────────────────────────────────────────────────────────────
def print_summary(data: dict, top_k: int = 20):
    corr_mean  = data["corr_mean"]
    corr_std   = data["corr_std"]
    corr_frac  = data["corr_frac_pos"]
    top_idx    = data["top_k_indices"]

    n_active = (corr_mean > 0.3).sum()
    n_consistent = ((corr_mean > 0.3) & (corr_frac > 0.8)).sum()

    print(f"\n{'='*60}")
    print(f"  SAE type : {data['sae_type']}")
    print(f"  Videos   : {data['features'].shape[0]}")
    print(f"  Features : {data['features'].shape[2]}")
    print(f"  Features with mean r > 0.3            : {n_active}")
    print(f"  Features with mean r > 0.3 AND frac>80%: {n_consistent}")
    print(f"{'='*60}")
    print(f"{'rank':>4}  {'feat':>6}  {'mean_r':>7}  {'std_r':>6}  {'frac>0':>7}")
    print(f"{'-'*40}")
    for rank, fidx in enumerate(top_idx[:top_k], 1):
        print(f"{rank:4d}  {fidx:6d}  {corr_mean[fidx]:7.4f}  "
              f"{corr_std[fidx]:6.4f}  {corr_frac[fidx]:7.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pkl", nargs="+", help="One or more .pkl result files")
    p.add_argument("--top_k",   type=int, default=9,
                   help="Number of top features to plot (default 9)")
    p.add_argument("--out_dir", default="figures",
                   help="Output directory for figures")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pkl_path in args.pkl:
        tag  = Path(pkl_path).stem     # e.g. "nonmono_std"
        data = load(pkl_path)
        print(f"\n{'#'*60}")
        print(f"  {pkl_path}")
        print_summary(data, top_k=args.top_k)

        plot_corr_distribution(data, out_dir, tag)
        plot_top_trajectories(data, out_dir, tag, top_k=args.top_k)
        plot_consistency_scatter(data, out_dir, tag, top_k=30)
        plot_heatmap(data, out_dir, tag, top_k=args.top_k)

    print(f"\nAll figures saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
