#!/bin/bash
#SBATCH --job-name=fit_pca_ica
#SBATCH --output=/home/renqy/logs/fit_pca_ica_%j.out
#SBATCH --error=/home/renqy/logs/fit_pca_ica_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/linear_decomp

cd ~/sae-for-vlm

# Fit PCA and ICA on the SAME VideoMAE layer-11 training corpus the SAE was trained
# on, so the linear-decomposition baselines are directly comparable to the SAE.
#   - TRAIN_ACT_DIR is the SAE training activations (jobs/dataset/run_extract_videomae_activations.sh)
#   - Outputs pca.pt and ica.pt under OUT_DIR (loaded later via PCADict/ICADict.from_pretrained)

TRAIN_ACT_DIR="/net/scratch/renqy/activations/train"
OUT_DIR="/net/scratch2/renqy/linear_decomp"

python analysis/fit_pca_ica.py \
    --activations_dir "${TRAIN_ACT_DIR}" \
    --output_dir      "${OUT_DIR}" \
    --n_components    768 \
    --n_samples       500000 \
    --max_chunks      20 \
    --methods         pca ica \
    --ica_max_iter    2000 \
    --ica_tol         1e-3 \
    --seed            0

# If ICA reports "DID NOT CONVERGE" in the log, re-run with fewer components
# (e.g. --n_components 256) or a larger --ica_max_iter.
