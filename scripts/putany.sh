#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate ts-rep
GPU_NUM=0

N_EXPERIMENTS=1

# Standardized
dataset_name="putany_std"
dataset_dir="datasets/putany/"
OUT_CHANNELS=100 #512 # commented ones are for IAE comparision
BATCH_SIZE=256 #64
CNN_OUTPUT_SIZE=80 #160


for (( i=1; i<=$N_EXPERIMENTS; i++ ))
do
    echo experiment $i
    python Putany.py \
    --dataset_name $dataset_name \
    --dataset_dir $dataset_dir \
    --gpu $GPU_NUM \
    --batch_size $BATCH_SIZE \
    --cnn_output_size $CNN_OUTPUT_SIZE \
    --out_channels $OUT_CHANNELS \
    --classification_task \
    --save_memory
done