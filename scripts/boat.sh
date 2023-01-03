#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate ts-rep
GPU_NUM=0

N_EXPERIMENTS=1

# Standardized and zero_padded
dataset_name="boat_std_zero_padded"
dataset_dir="datasets/boat/fixed/"

# Standardized and nan padded for varying length
# dataset_name="boat_std_nan_padded"
# dataset_dir="datasets/boat/varying/"


for (( i=1; i<=$N_EXPERIMENTS; i++ ))
do
    echo experiment $i
    python Boat.py \
    --dataset_name $dataset_name \
    --dataset_dir $dataset_dir \
    --gpu $GPU_NUM \
    --clustering_task \
    --anomaly_detection_task #\
    # --save_memory
done