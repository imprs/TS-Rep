#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate ts-rep
GPU_NUM=0

N_EXPERIMENTS=1

# Standardized and zero_padded
dataset_name="qcat_std_zero_padded"
dataset_dir="datasets/qcat/fixed/"
MODALITY="force"
BATCH_SIZE=256


for (( i=1; i<=$N_EXPERIMENTS; i++ ))
do
    echo experiment $i
    python Qcat.py \
    --dataset_name $dataset_name \
    --dataset_dir $dataset_dir \
    --batch_size $BATCH_SIZE \
    --modality $MODALITY \
    --gpu $GPU_NUM \
    --classification_task \
    --save_memory

done