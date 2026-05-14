#!/bin/bash
#SBATCH --job-name=ablation_analysis
#SBATCH --output=/net/scratch/renqy/logs/ablation_analysis_%j.out
#SBATCH --error=/net/scratch/renqy/logs/ablation_analysis_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

set -euo pipefail

cd ~/sae-for-vlm

SAE_DIR="/net/scratch/renqy/sae_checkpoints_10ep"
BALL_DIR=~/sae-for-vlm/ball_dataset
OUT_DIR="/net/scratch/renqy/ball_features"
mkdir -p /net/scratch/renqy/logs "${OUT_DIR}"

MODEL="MCG-NJU/videomae-base-finetuned-ssv2"

for DATASET_TYPE in static backforth; do
  for SAE_TYPE in batch_top_k standard; do
    if [ "${SAE_TYPE}" = "batch_top_k" ]; then
      SAE_PATH="${SAE_DIR}/train_batch_top_k_64_x8/trainer_0/ae.pt"
      TAG="btk"
    else
      SAE_PATH="${SAE_DIR}/train_standard_8_x8/trainer_0/ae.pt"
      TAG="std"
    fi

    echo "=== ${DATASET_TYPE} / ${SAE_TYPE} ==="
    python analysis/analyze_ablation.py \
      --dataset_type  "${DATASET_TYPE}" \
      --dataset_dir   "${BALL_DIR}/${DATASET_TYPE}" \
      --sae_path      "${SAE_PATH}" \
      --sae_type      "${SAE_TYPE}" \
      --output_path   "${OUT_DIR}/${DATASET_TYPE}_${TAG}.pkl" \
      --model_name    "${MODEL}" \
      --layer         11 \
      --top_k         100 \
      --batch_size    8 \
      --num_workers   4 \
      --device        cuda:0
  done
done

echo "All done. Results in ${OUT_DIR}/"
