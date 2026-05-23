#!/bin/bash
#SBATCH --job-name=extract_videomae
#SBATCH --output=/home/renqy/logs/extract_videomae_%j.out
#SBATCH --error=/home/renqy/logs/extract_videomae_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs \
         /net/scratch/renqy/activations/train \
         /net/scratch/renqy/activations/val

cd ~/sae-for-vlm

MODEL="MCG-NJU/videomae-base-finetuned-ssv2"
DATA_PATH="/net/scratch/renqy/SSv2"

# Train split — all tokens per video (no pooling), used for SAE training
python save_activations.py \
    --model_name       "${MODEL}" \
    --attachment_point post_mlp_residual \
    --layer            11 \
    --dataset_name     ssv2 \
    --data_path        "${DATA_PATH}" \
    --split            train \
    --batch_size       8 \
    --num_workers      8 \
    --output_dir       /net/scratch/renqy/activations/train \
    --device           cuda:0

# Val split
python save_activations.py \
    --model_name       "${MODEL}" \
    --attachment_point post_mlp_residual \
    --layer            11 \
    --dataset_name     ssv2 \
    --data_path        "${DATA_PATH}" \
    --split            val \
    --batch_size       8 \
    --num_workers      8 \
    --output_dir       /net/scratch/renqy/activations/val \
    --device           cuda:0
