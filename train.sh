#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
task="textvqa"
model="llava_new"
method="rel_att"
max_sample=-1
batch_size=6
lora_name=${task}-"trainhtl-0508"

# 构造日志文件名
log_file="${task}-${CUDA_VISIBLE_DEVICES}.log"

# 追加写入 LoRA 名称及时间戳
echo -e "\n===== New Run =====\nLoRA name: $lora_name\nStart time: $(date)\n" >> $log_file

# 后台训练过程日志也追加写入
python _train.py \
    --task $task \
    --model $model \
    --max_sample $max_sample \
    --epochs 3 \
    --lora_name $lora_name \
    --batch_size $batch_size \
    --is_eval 1 
    # >> $log_file 2>&1 &
