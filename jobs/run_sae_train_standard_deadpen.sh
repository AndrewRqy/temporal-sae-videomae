#!/bin/bash
#SBATCH --job-name=sae_standard_deadpen
#SBATCH --output=/home/renqy/logs/sae_standard_deadpen_%j.out
#SBATCH --error=/home/renqy/logs/sae_standard_deadpen_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /net/scratch/renqy/logs /net/scratch/renqy/sae_checkpoints_deadpen_0p03

cd ~/sae-for-vlm

python sae_train.py \
    --sae_model standard \
    --activations_dir /net/scratch/renqy/activations/train \
    --val_activations_dir /net/scratch/renqy/activations/val \
    --checkpoints_dir /net/scratch/renqy/sae_checkpoints_deadpen_0p03 \
    --expansion_factor 8 \
    --l1_penalty 0.1 \
    --auxk_alpha 0.0 \
    --dead_penalty_coef 0.03 \
    --batch_size 4096 \
    --steps 15320 \
    --save_steps 7660 \
    --log_steps 500 \
    --lr 2e-4 \
    --wandb_entity relay-andrew2020-the-university-of-chicago \
    --wandb_project "VideoMAE SAE" \
    --device cuda:0
