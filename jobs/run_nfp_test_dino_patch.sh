#!/bin/bash
#SBATCH --job-name=nfp_dino_patch
#SBATCH --output=/home/renqy/logs/nfp_dino_patch_%j.out
#SBATCH --error=/home/renqy/logs/nfp_dino_patch_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/nfp_results

cd ~/sae-for-vlm

python analysis/nfp_test_dino_patch.py \
    --dataset_dir  /net/scratch2/renqy/nfp \
    --sae_path     /net/scratch2/renqy/sae_checkpoints_dino_patch/train_standard_8_x8/trainer_0/ae.pt \
    --output_path  /net/scratch2/renqy/nfp_results/dino_patch_nfp.pt \
    --batch_size   8 \
    --num_workers  4 \
    --device       cuda:0
