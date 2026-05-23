#!/bin/bash
#SBATCH --job-name=save_dino_patch
#SBATCH --output=/home/renqy/logs/save_dino_patch_%j.out
#SBATCH --error=/home/renqy/logs/save_dino_patch_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs \
         /net/scratch2/renqy/dino_patch_activations/train \
         /net/scratch2/renqy/dino_patch_activations/val

cd ~/sae-for-vlm

SSV2_PATH="/net/scratch/renqy/SSv2"

# Train split — 4000 videos x 16 frames x 196 tokens = 12.544M patch tokens
python training/extract_dino_patch_activations.py \
    --data_path    "${SSV2_PATH}" \
    --output_dir   /net/scratch2/renqy/dino_patch_activations/train \
    --split        train \
    --max_videos   4000 \
    --batch_size   8 \
    --save_every   50000 \
    --num_workers  8 \
    --device       cuda:0

# Val split — 800 videos x 16 frames x 196 tokens = 2.508M patch tokens
python training/extract_dino_patch_activations.py \
    --data_path    "${SSV2_PATH}" \
    --output_dir   /net/scratch2/renqy/dino_patch_activations/val \
    --split        val \
    --max_videos   800 \
    --batch_size   8 \
    --save_every   50000 \
    --num_workers  8 \
    --device       cuda:0
