#!/bin/bash
#SBATCH --job-name=ssv2_dino_embed
#SBATCH --output=/home/renqy/logs/ssv2_dino_embed_%j.out
#SBATCH --error=/home/renqy/logs/ssv2_dino_embed_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch/renqy/embeddings

cd ~/sae-for-vlm

# DINOv2-base embeddings for SSv2 val split (max-pooled over 16 frames per clip).
# Used as the vision encoder for monosemanticity scoring.
python eval/encode_videos.py \
    --embeddings_path /net/scratch/renqy/embeddings/ssv2_val_dinov2.pt \
    --model_name      dinov2-base \
    --data_path       /net/scratch/renqy/SSv2 \
    --split           val \
    --batch_size      32 \
    --num_workers     8 \
    --max_clips       800 \
    --device          cuda:0
