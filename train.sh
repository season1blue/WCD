#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
task="mvsa_s"
model="llava"
method="rel_att"
max_sample=10
batch_size=4
lora_name=${task}-"withadalossmlpx-0430"

python _train.py \
    --task $task \
    --model $model \
    --max_sample $max_sample \
    --epochs 10 \
    --lora_name $lora_name \
    --batch_size $batch_size \
    --is_eval 1