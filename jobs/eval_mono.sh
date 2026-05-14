#!/bin/bash
#SBATCH --job-name=mono_score_videomae
#SBATCH --output=/net/scratch/renqy/logs/mono_score_%j.out
#SBATCH --error=/net/scratch/renqy/logs/mono_score_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

cd ~/sae-for-vlm

MODEL="MCG-NJU/videomae-base-finetuned-ssv2"
DATA_PATH="/net/scratch/renqy/SSv2"
LAYER=11
POINT="post_mlp_residual"
MAX_CLIPS=800
BATCH=8
WORKERS=4
DEVICE="cuda:0"
SAE_DIR="/net/scratch/renqy/sae_checkpoints_10ep"
ACT_DIR="/net/scratch/renqy/sae_activations"
EMBED_PATH="/net/scratch/renqy/embeddings/ssv2_val_dinov2.pt"

# Step 3a: SAE activations â€?batch_top_k with max-pool
python training/extract_activations.py \
    --model_name "${MODEL}" \
    --attachment_point "${POINT}" \
    --layer "${LAYER}" \
    --dataset_name ssv2 \
    --data_path "${DATA_PATH}" \
    --split val \
    --batch_size "${BATCH}" \
    --num_workers "${WORKERS}" \
    --output_dir "${ACT_DIR}/batch_top_k_val" \
    --max_pool \
    --sae_model batch_top_k \
    --sae_path "${SAE_DIR}/train_batch_top_k_64_x8/trainer_0/ae.pt" \
    --max_clips "${MAX_CLIPS}" \
    --device "${DEVICE}"

# Step 3b: SAE activations â€?standard SAE with max-pool
python training/extract_activations.py \
    --model_name "${MODEL}" \
    --attachment_point "${POINT}" \
    --layer "${LAYER}" \
    --dataset_name ssv2 \
    --data_path "${DATA_PATH}" \
    --split val \
    --batch_size "${BATCH}" \
    --num_workers "${WORKERS}" \
    --output_dir "${ACT_DIR}/standard_val" \
    --max_pool \
    --sae_model standard \
    --sae_path "${SAE_DIR}/train_standard_8_x8/trainer_0/ae.pt" \
    --max_clips "${MAX_CLIPS}" \
    --device "${DEVICE}"

# Step 3c: Raw activations (baseline) with max-pool
python training/extract_activations.py \
    --model_name "${MODEL}" \
    --attachment_point "${POINT}" \
    --layer "${LAYER}" \
    --dataset_name ssv2 \
    --data_path "${DATA_PATH}" \
    --split val \
    --batch_size "${BATCH}" \
    --num_workers "${WORKERS}" \
    --output_dir "${ACT_DIR}/raw_val" \
    --max_pool \
    --max_clips "${MAX_CLIPS}" \
    --device "${DEVICE}"

# Step 4: DINOv2 embeddings (max-pooled over 16 frames per clip)
python eval/encode_videos.py \
    --embeddings_path "${EMBED_PATH}" \
    --model_name dinov2-base \
    --data_path "${DATA_PATH}" \
    --split val \
    --batch_size 32 \
    --num_workers "${WORKERS}" \
    --max_clips "${MAX_CLIPS}" \
    --device "${DEVICE}"

# Step 5: Monosemanticity score
python eval/metric.py \
    --activations_dir "${ACT_DIR}/batch_top_k_val" \
    --embeddings_path "${EMBED_PATH}" \
    --output_subdir ms_dinov2

python eval/metric.py \
    --activations_dir "${ACT_DIR}/standard_val" \
    --embeddings_path "${EMBED_PATH}" \
    --output_subdir ms_dinov2

python eval/metric.py \
    --activations_dir "${ACT_DIR}/raw_val" \
    --embeddings_path "${EMBED_PATH}" \
    --output_subdir ms_dinov2
