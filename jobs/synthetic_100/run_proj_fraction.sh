#!/bin/bash
#SBATCH --job-name=proj_fraction
#SBATCH --output=/home/renqy/logs/proj_fraction_%j.out
#SBATCH --error=/home/renqy/logs/proj_fraction_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs

cd ~/sae-for-vlm

# Prerequisite: run run_nfp_test_synthetic.sh first (needs synthetic_nfp.pt)
python analysis/proj_fraction.py \
    --sae_path      /net/scratch2/renqy/sae_checkpoints_synthetic/train_standard_8_x8/trainer_0/ae.pt \
    --matrices_path /net/scratch2/renqy/synthetic_acts/matrices.pt \
    --nfp_results   /net/scratch2/renqy/nfp_results/synthetic_nfp.pt \
    --device        cpu
