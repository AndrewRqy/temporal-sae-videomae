#!/bin/bash
#SBATCH --job-name=mono_score_dino
#SBATCH --output=/home/renqy/logs/mono_score_dino_%j.out
#SBATCH --error=/home/renqy/logs/mono_score_dino_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/sae_activations

cd ~/sae-for-vlm

SSV2_PATH="/net/scratch/renqy/SSv2"
MODEL="dinov2-base"
DEVICE="cuda:0"
EMBED_PATH="/net/scratch/renqy/embeddings/ssv2_val_dinov2.pt"
SAE_PATH="/net/scratch2/renqy/sae_checkpoints_dino/train_standard_8_x8/trainer_0/ae.pt"
DINO_SAE_ACT_DIR="/net/scratch2/renqy/sae_activations/dino_sae_val"

# Step 1: Extract video-level DINOv2 SAE activations on SSv2 val
python encode_dino_sae_videos.py \
    --output_dir "${DINO_SAE_ACT_DIR}" \
    --sae_path "${SAE_PATH}" \
    --model_name "${MODEL}" \
    --data_path "${SSV2_PATH}" \
    --split val \
    --batch_size 32 \
    --num_workers 8 \
    --save_every 5000 \
    --device "${DEVICE}"

# Step 2: DINOv2 video embeddings for SSv2 val (already computed, skips if exists)
python encode_videos.py \
    --embeddings_path "${EMBED_PATH}" \
    --model_name "${MODEL}" \
    --data_path "${SSV2_PATH}" \
    --split val \
    --batch_size 32 \
    --num_workers 8 \
    --device "${DEVICE}"

# Step 3: Monosemanticity score
python metric.py \
    --activations_dir "${DINO_SAE_ACT_DIR}" \
    --embeddings_path "${EMBED_PATH}" \
    --output_subdir ms_dinov2
