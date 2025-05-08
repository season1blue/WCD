#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
task="textvqa"
model="llava_new"
method="rel_att"
max_sample=10
batch_size=4
lora_name=${task}-"train-0508"

python _train.py \
    --task $task \
    --model $model \
    --max_sample $max_sample \
    --epochs 3 \
    --lora_name $lora_name \
    --batch_size $batch_size \
    --is_eval 1