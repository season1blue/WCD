#!/bin/bash
# 用法：nohup bash eval_queue.sh > global_log.txt 2>&1 &

#############################################
#             通用函数定义
#############################################

# 获取最空闲的 GPU
get_best_gpu() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk '{print $1}' \
    | awk 'BEGIN{max=-1; idx=-1} {if ($1+0 > max) {max=$1; idx=NR-1}} END{print idx}'
}

# 检查 GPU 是否空闲（空闲显存 > threshold）
is_gpu_free() {
    local gpu_id=$1
    local threshold=${2:-40000}  # 默认阈值：显存空余大于 40GB 视为空闲
    local free_mem
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((gpu_id+1))p")
    if [ "$free_mem" -gt "$threshold" ]; then
        return 0  # 空闲
    else
        return 1  # 忙碌
    fi
}

#############################################
#             模型与任务设置
#############################################

model="llavaonevision"
method="rel_att"
max_sample=1000
batch_size=1
attn_layer_idx=17
mask_ratio=1
log_output=false

#############################################
#             主任务循环
#############################################

for task in gqa
do
    for target_layer_idx in {20..28}
    do
        # 查找空闲 GPU
        while true; do
            GPU_ID=$(get_best_gpu)
            if is_gpu_free "$GPU_ID" 40000; then
                export CUDA_VISIBLE_DEVICES=$GPU_ID
                echo "[INFO] Selected GPU $GPU_ID (free memory OK)"
                break
            else
                echo "[WAIT] All GPUs busy... checking again in 60s"
                sleep 60
            fi
        done

        # 设置运行参数
        result_path="result_${CUDA_VISIBLE_DEVICES}.json"
        lora_name=$task-$target_layer_idx-$model-$mask_ratio
        log_path="log/$model/log_$task.log"

        cmd="python _main.py \
            --model $model \
            --task $task \
            --max_sample $max_sample \
            --lora_name $lora_name \
            --batch_size $batch_size \
            --attn_layer_idx $attn_layer_idx \
            --target_layer_idx $target_layer_idx \
            --result_path $result_path \
            --mask_ratio $mask_ratio"

        echo "[RUNNING] GPU=$GPU_ID | task=$task | layer=$target_layer_idx"

        if [ "$log_output" = true ]; then
            nohup bash -c "$cmd" >> "$log_path" 2>&1 &
        else
            eval "$cmd"
        fi

        echo "[DONE] layer=$target_layer_idx finished"
        sleep 5  # 稍作休息防止冲突
    done
done
