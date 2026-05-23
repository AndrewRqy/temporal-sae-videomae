#!/bin/bash
#SBATCH --job-name=gen_synth_763
#SBATCH --output=/home/renqy/logs/gen_synth_763_%j.out
#SBATCH --error=/home/renqy/logs/gen_synth_763_%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/synthetic_acts_763

cd ~/sae-for-vlm

# 768 - 5 = 763 static directions: temporal + static span the full R^768
# Prerequisite: NFP ball videos at /net/scratch2/renqy/nfp
python analysis/gen_synthetic_activations.py \
    --nfp_dir     /net/scratch2/renqy/nfp \
    --output_dir  /net/scratch2/renqy/synthetic_acts_763 \
    --dim         768 \
    --n_static    763 \
    --noise_scale 0.05 \
    --seed        42 \
    --val_frac    0.2
