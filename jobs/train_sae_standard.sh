#!/bin/bash
#SBATCH --job-name=sae_train_standard
#SBATCH --output=/net/scratch/renqy/logs/sae_train_standard_%j.out
#SBATCH --error=/net/scratch/renqy/logs/sae_train_standard_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

cd ~/sae-for-vlm

python training/train_sae.py \
    --sae_model standard \
    --activations_dir /net/scratch/renqy/activations/train \
    --val_activations_dir /net/scratch/renqy/activations/val \
    --checkpoints_dir /net/scratch/renqy/sae_checkpoints_10ep \
    --expansion_factor 8 \
    --l1_penalty 0.1 \
    --batch_size 4096 \
    --steps 15320 \
    --save_steps 1532 \
    --log_steps 50 \
    --lr 2e-4 \
    --wandb_entity relay-andrew2020-the-university-of-chicago \
    --wandb_project "VideoMAE SAE" \
    --device cuda:0
