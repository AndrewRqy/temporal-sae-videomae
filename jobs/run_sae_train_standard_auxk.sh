#!/bin/bash
#SBATCH --job-name=sae_standard_auxk
#SBATCH --output=/net/scratch/renqy/logs/sae_standard_auxk_%j.out
#SBATCH --error=/net/scratch/renqy/logs/sae_standard_auxk_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /net/scratch/renqy/logs /net/scratch/renqy/sae_checkpoints_auxk

cd ~/sae-for-vlm

python sae_train.py \
    --sae_model standard \
    --activations_dir /net/scratch/renqy/activations/train \
    --val_activations_dir /net/scratch/renqy/activations/val \
    --checkpoints_dir /net/scratch/renqy/sae_checkpoints_auxk_low_thresh \
    --expansion_factor 8 \
    --l1_penalty 0.1 \
    --auxk_alpha 0.03125 \
    --dead_feature_threshold 200000 \
    --batch_size 4096 \
    --steps 15320 \
    --save_steps 1532 \
    --log_steps 50 \
    --lr 2e-4 \
    --wandb_entity relay-andrew2020-the-university-of-chicago \
    --wandb_project "VideoMAE SAE" \
    --device cuda:0
