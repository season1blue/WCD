#!/bin/bash

# nohup bash eval.sh > global_log.txt 2>&1 &

get_best_gpu() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk '{print $1}' \
    | awk 'BEGIN{max=-1; idx=-1} {if ($1+0 > max) {max=$1; idx=NR-1}} END{print idx}'
}

GPU_ID=$(get_best_gpu)
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU $CUDA_VISIBLE_DEVICES"

task="textvqa"
model="llava_new"
method="rel_att"
max_sample=1000
batch_size=1
attn_layer_idx=17
lora_name=$task

result_path="result_${CUDA_VISIBLE_DEVICES}.json"
log_path="log_${task}.log"

for target_layer_idx in {2..31}
do
    echo "Running target_layer_idx=$target_layer_idx" >> "$log_path"

    cmd="python _eval.py \
        --model $model \
        --task $task \
        --max_sample $max_sample \
        --lora_name $lora_name \
        --batch_size $batch_size \
        --attn_layer_idx $attn_layer_idx \
        --target_layer_idx $target_layer_idx \
        --result_path $result_path \
        "
    echo "Executing: $cmd" >> "$log_path"

    eval $cmd >> "$log_path" 2>&1
    
    sleep 1  # 避免瞬间启动多个任务造成冲突
done
