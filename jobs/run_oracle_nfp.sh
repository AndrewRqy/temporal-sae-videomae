#!/bin/bash
#SBATCH --job-name=oracle_nfp
#SBATCH --output=/home/renqy/logs/oracle_nfp_%j.out
#SBATCH --error=/home/renqy/logs/oracle_nfp_%j.err
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
ORACLE_PATH="/net/scratch2/renqy/nfp_results/oracle_ae.pt"
NFP_DIR="/net/scratch2/renqy/nfp"
OUT_DIR="/net/scratch2/renqy/nfp_results"

python analysis/nfp_test.py \
    --dataset_dir "${NFP_DIR}" \
    --sae_path    "${ORACLE_PATH}" \
    --output_path "${OUT_DIR}/oracle_nfp.pt" \
    --model_name  "${MODEL}" \
    --layer       11 \
    --attachment_point post_mlp_residual \
    --batch_size  8 \
    --num_workers 8 \
    --label       "Oracle SAE (positive control)" \
    --device      cuda:0
