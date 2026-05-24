#!/bin/bash
#SBATCH --job-name=nfp_avgtau
#SBATCH --output=/home/renqy/logs/nfp_avgtau_%j.out
#SBATCH --error=/home/renqy/logs/nfp_avgtau_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/nfp_results

cd ~/sae-for-vlm

python analysis/nfp_test.py \
    --dataset_dir   /net/scratch2/renqy/nfp \
    --sae_path      /net/scratch/renqy/sae_checkpoints_deadpen_0p03/train_standard_8_x8/trainer_0/ae.pt \
    --output_path   /net/scratch2/renqy/nfp_results/videomae_deadpen_0p03_avgtau.pt \
    --layer         11 \
    --tau_mode      avg_frames \
    --label         "VideoMAE SAE (avg-tau)" \
    --batch_size    4 \
    --num_workers   4 \
    --device        cuda:0
