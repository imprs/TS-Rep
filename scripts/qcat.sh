#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate ts-rep
GPU_NUM=0

N_EXPERIMENTS=1

# Standardized
dataset_name="qcat_std_zero_padded_speed_66"
dataset_dir="datasets/qcat_6/"


for (( i=1; i<=$N_EXPERIMENTS; i++ ))
do
    echo experiment $i
    python Qcat.py \
    --dataset_name $dataset_name \
    --dataset_dir $dataset_dir \
    --gpu $GPU_NUM \
    --clustering_task \
    --anomaly_detection_task #\
    # --save_memory
done