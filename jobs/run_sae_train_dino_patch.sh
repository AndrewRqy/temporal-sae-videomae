#!/bin/bash
#SBATCH --job-name=sae_dino_patch
#SBATCH --output=/home/renqy/logs/sae_dino_patch_%j.out
#SBATCH --error=/home/renqy/logs/sae_dino_patch_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/sae_checkpoints_dino_patch

cd ~/sae-for-vlm

# ~50 train chunks x 50k tokens = 2.5M tokens, 1 val chunk (50k tokens)
# batch_size=4096 → ~610 steps/epoch; 15,320 steps ≈ 25 epochs (matches VideoMAE SAE)
# Note: val dir contains a single chunk copied from train after disk-quota cleanup.
python sae_train.py \
    --sae_model           standard \
    --activations_dir     /net/scratch2/renqy/dino_patch_activations/train \
    --val_activations_dir /net/scratch2/renqy/dino_patch_activations/val \
    --checkpoints_dir     /net/scratch2/renqy/sae_checkpoints_dino_patch \
    --expansion_factor    8 \
    --l1_penalty          0.1 \
    --auxk_alpha          0.0 \
    --dead_penalty_coef   0.03 \
    --batch_size          4096 \
    --steps               15320 \
    --save_steps          7660 \
    --log_steps           500 \
    --lr                  2e-4 \
    --no_wandb \
    --device              cuda:0
