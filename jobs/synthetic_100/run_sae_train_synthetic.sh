#!/bin/bash
#SBATCH --job-name=sae_synthetic
#SBATCH --output=/home/renqy/logs/sae_synthetic_%j.out
#SBATCH --error=/home/renqy/logs/sae_synthetic_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/sae_checkpoints_synthetic

cd ~/sae-for-vlm

python sae_train.py \
    --sae_model           standard \
    --activations_dir     /net/scratch2/renqy/synthetic_acts/train \
    --val_activations_dir /net/scratch2/renqy/synthetic_acts/val \
    --checkpoints_dir     /net/scratch2/renqy/sae_checkpoints_synthetic \
    --expansion_factor    8 \
    --l1_penalty          0.1 \
    --auxk_alpha          0.0 \
    --dead_penalty_coef   0.03 \
    --batch_size          4096 \
    --steps               5000 \
    --save_steps          2500 \
    --log_steps           200 \
    --lr                  2e-4 \
    --no_wandb \
    --device              cuda:0
