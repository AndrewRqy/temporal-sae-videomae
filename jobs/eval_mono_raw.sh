#!/bin/bash
#SBATCH --job-name=mono_raw
#SBATCH --output=/home/renqy/logs/mono_raw_%j.out
#SBATCH --error=/home/renqy/logs/mono_raw_%j.err
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

# Extract raw VideoMAE layer-11 activations (no SAE, max-pooled over tokens)
python save_activations.py \
    --model_name "${MODEL}" \
    --attachment_point "${POINT}" \
    --layer "${LAYER}" \
    --dataset_name ssv2 \
    --data_path "${DATA_PATH}" \
    --split val \
    --batch_size "${BATCH}" \
    --num_workers "${WORKERS}" \
    --output_dir "${ACT_DIR}/raw_val" \
    --max_pool \
    --max_clips "${MAX_CLIPS}" \
    --device "${DEVICE}"

# Monosemanticity score on raw activations
# DINOv2 embeddings already exist at EMBED_PATH from prior eval_mono run
python eval/metric.py \
    --activations_dir "${ACT_DIR}/raw_val" \
    --embeddings_path "${EMBED_PATH}" \
    --output_subdir ms_dinov2
