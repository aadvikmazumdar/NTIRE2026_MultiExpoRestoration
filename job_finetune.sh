#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
source ~/.bashrc
conda activate RawFusion
cd /mnt/home2/home/nirmala_b/RawFusion
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_finetuning.py
