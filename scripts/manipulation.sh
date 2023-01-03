#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ts-rep
GPU_NUM=0

N_EXPERIMENTS=1

# Standardized and resampled
dataset_name="manipulation_std_resampled"
dataset_dir="datasets/manipulation/fixed/"

# Standardized and nan padded for varying length
# dataset_name="manipulation_std_nan_padded"
# dataset_dir="datasets/manipulation/varying"

for (( i=1; i<=$N_EXPERIMENTS; i++ ))
do
    echo experiment $i
    python Manipulation.py \
    --dataset_name $dataset_name \
    --dataset_dir $dataset_dir \
    --gpu $GPU_NUM \
    --clustering_task \
    --anomaly_detection_task #\
    # --save_memory
done