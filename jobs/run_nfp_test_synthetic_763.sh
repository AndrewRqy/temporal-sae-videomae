#!/bin/bash
#SBATCH --job-name=nfp_synth_763
#SBATCH --output=/home/renqy/logs/nfp_synth_763_%j.out
#SBATCH --error=/home/renqy/logs/nfp_synth_763_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/nfp_results

cd ~/sae-for-vlm

SYNTH_DIR="/net/scratch2/renqy/synthetic_acts_763"
SAE_PATH="/net/scratch2/renqy/sae_checkpoints_synthetic_763/train_standard_8_x8/trainer_0/ae.pt"

python analysis/nfp_test_synthetic.py \
    --all_videos_path "${SYNTH_DIR}/all_videos.pt" \
    --matrices_path   "${SYNTH_DIR}/matrices.pt" \
    --sae_path        "${SAE_PATH}" \
    --output_path     /net/scratch2/renqy/nfp_results/synthetic_763_nfp.pt \
    --device          cuda:0
