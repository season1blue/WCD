#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
task="textvqa"
model="llava"
method="rel_att"
max_sample=500
batch_size=4
lora_name=${task}-"base-0506"

python _train.py \
    --task $task \
    --model $model \
    --max_sample $max_sample \
    --epochs 3 \
    --lora_name $lora_name \
    --batch_size $batch_size \
    --is_eval 1