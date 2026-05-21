#!/bin/bash
#SBATCH --job-name=nfp_test_dino
#SBATCH --output=/home/renqy/logs/nfp_test_dino_%j.out
#SBATCH --error=/home/renqy/logs/nfp_test_dino_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/nfp_results

cd ~/sae-for-vlm

SAE_PATH="/net/scratch2/renqy/sae_checkpoints_dino/train_standard_8_x8/trainer_0/ae.pt"
NFP_DIR="/net/scratch2/renqy/nfp"
OUTPUT="/net/scratch2/renqy/nfp_results/dino_negative_control.pt"

python analysis/nfp_test_dino.py \
    --dataset_dir "${NFP_DIR}" \
    --sae_path    "${SAE_PATH}" \
    --output_path "${OUTPUT}" \
    --model_name  dinov2-base \
    --batch_size  8 \
    --num_workers 8 \
    --device      cuda:0
