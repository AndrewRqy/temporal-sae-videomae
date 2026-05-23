#!/bin/bash
#SBATCH --job-name=build_oracle
#SBATCH --output=/home/renqy/logs/build_oracle_%j.out
#SBATCH --error=/home/renqy/logs/build_oracle_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=k003

export PATH=/home/renqy/.conda/envs/ttic31280/bin:$PATH

mkdir -p /home/renqy/logs /net/scratch2/renqy/nfp_results

cd ~/sae-for-vlm

MODEL="MCG-NJU/videomae-base-finetuned-ssv2"
SAE_PATH="/net/scratch/renqy/sae_checkpoints_deadpen_0p03/train_standard_8_x8/trainer_0/ae.pt"
NFP_DIR="/net/scratch2/renqy/nfp"
ORACLE_PATH="/net/scratch2/renqy/nfp_results/oracle_ae.pt"

python analysis/build_oracle_sae.py \
    --dataset_dir "${NFP_DIR}" \
    --sae_path    "${SAE_PATH}" \
    --output_path "${ORACLE_PATH}" \
    --model_name  "${MODEL}" \
    --layer       11 \
    --attachment_point post_mlp_residual \
    --batch_size  8 \
    --num_workers 8 \
    --device      cuda:0
