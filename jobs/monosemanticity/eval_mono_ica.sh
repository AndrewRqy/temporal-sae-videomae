#!/bin/bash
#SBATCH --job-name=mono_score_ica
#SBATCH --output=/home/renqy/logs/mono_score_ica_%j.out
#SBATCH --error=/home/renqy/logs/mono_score_ica_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/sae_activations

cd ~/sae-for-vlm

MODEL="MCG-NJU/videomae-base-finetuned-ssv2"
DATA_PATH="/net/scratch/renqy/SSv2"
LAYER=11
POINT="post_mlp_residual"
MAX_CLIPS=800
BATCH=8
WORKERS=4
DEVICE="cuda:0"
ACT_DIR="/net/scratch2/renqy/sae_activations"
EMBED_PATH="/net/scratch/renqy/embeddings/ssv2_val_dinov2.pt"
ICA_PATH="/net/scratch2/renqy/linear_decomp/ica.pt"

# Sign-handling mode for signed ICA components:
#   sign_split (primary) -> ReLU(+c)/ReLU(-c), 768 -> 1536 features
#   abs        (robustness) -> |c|, 768 features
MODE="${1:-sign_split}"

# Prerequisites:
#   - DINOv2 embeddings at EMBED_PATH (jobs/dataset/run_ssv2_dino_embeddings.sh)
#   - Fitted ICA at ICA_PATH         (jobs/monosemanticity/fit_pca_ica.sh)
# Same MODEL / LAYER / POINT / clips / EMBED_PATH as eval_mono_auxk.sh and
# eval_mono_raw.sh, so SAE / raw / PCA / ICA scores are directly comparable.

# Extract max-pooled ICA-component activations (one vector per clip)
python save_activations.py \
    --model_name       "${MODEL}" \
    --attachment_point "${POINT}" \
    --layer            "${LAYER}" \
    --dataset_name     ssv2 \
    --data_path        "${DATA_PATH}" \
    --split            val \
    --batch_size       "${BATCH}" \
    --num_workers      "${WORKERS}" \
    --output_dir       "${ACT_DIR}/ica_${MODE}_val" \
    --max_pool \
    --sae_model        ica \
    --sae_path         "${ICA_PATH}" \
    --decomp_mode      "${MODE}" \
    --max_clips        "${MAX_CLIPS}" \
    --device           "${DEVICE}"

# Monosemanticity score
python eval/metric.py \
    --activations_dir "${ACT_DIR}/ica_${MODE}_val" \
    --embeddings_path "${EMBED_PATH}" \
    --output_subdir   ms_dinov2
