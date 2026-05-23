#!/bin/bash
#SBATCH --job-name=gen_synthetic
#SBATCH --output=/home/renqy/logs/gen_synthetic_%j.out
#SBATCH --error=/home/renqy/logs/gen_synthetic_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/synthetic_acts

cd ~/sae-for-vlm

python analysis/gen_synthetic_activations.py \
    --nfp_dir    /net/scratch2/renqy/nfp \
    --output_dir /net/scratch2/renqy/synthetic_acts \
    --dim        768 \
    --n_static   100 \
    --noise_scale 0.05 \
    --seed       42 \
    --val_frac   0.2
