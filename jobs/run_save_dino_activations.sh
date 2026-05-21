#!/bin/bash
#SBATCH --job-name=save_dino_act
#SBATCH --output=/home/renqy/logs/save_dino_act_%j.out
#SBATCH --error=/home/renqy/logs/save_dino_act_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs \
         /net/scratch2/renqy/dino_activations/train \
         /net/scratch2/renqy/dino_activations/val

cd ~/sae-for-vlm

SSV2_PATH="/net/scratch/renqy/SSv2"
MODEL="dinov2-base"
LAYER=-1
POINT="pooler_output"
DEVICE="cuda:0"
TRAIN_CLIPS=64000   # 4000 videos x 16 frames (max_clips counts frames for DINOv2)
VAL_CLIPS=12800     # 800 videos x 16 frames

# Train split
python save_activations.py \
    --model_name "${MODEL}" \
    --attachment_point "${POINT}" \
    --layer "${LAYER}" \
    --dataset_name ssv2_dino \
    --data_path "${SSV2_PATH}" \
    --split train \
    --batch_size 32 \
    --num_workers 8 \
    --output_dir /net/scratch2/renqy/dino_activations/train \
    --save_every 50000 \
    --max_clips "${TRAIN_CLIPS}" \
    --device "${DEVICE}"

# Val split
python save_activations.py \
    --model_name "${MODEL}" \
    --attachment_point "${POINT}" \
    --layer "${LAYER}" \
    --dataset_name ssv2_dino \
    --data_path "${SSV2_PATH}" \
    --split val \
    --batch_size 32 \
    --num_workers 8 \
    --output_dir /net/scratch2/renqy/dino_activations/val \
    --save_every 50000 \
    --max_clips "${VAL_CLIPS}" \
    --device "${DEVICE}"
