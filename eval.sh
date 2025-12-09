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



# llava_new_1.6_7
model="qwen2.5"
method="rel_att"
max_sample=1000
batch_size=1
attn_layer_idx=0
result_path="result_${CUDA_VISIBLE_DEVICES}.json"


# timestamp=$(date +%Y%m%d_%H%M%S)
# tmp_py="tmp_eval_${timestamp}.py"
# cp _eval.py "$tmp_py"

# gqa vqav2 okvqa vizwiz textvqa docvqa

mask_ratio=1
# 1 2 32 33 34 35 36 37 38 39

log_output=false
for task in textvqa
    do
    # for target_layer_idx in {1..31}
    for target_layer_idx in 31
    # for ((target_layer_idx=31; target_layer_idx>=0; target_layer_idx--))
    do
        lora_name=$task-$target_layer_idx-$model-$mask_ratio
        log_path="log/$model/log_$task.log"
        mkdir -p "$(dirname "$log_path")"

        cmd="python _main.py  --model $model --task $task  --max_sample $max_sample --lora_name $lora_name --batch_size $batch_size --attn_layer_idx $attn_layer_idx --target_layer_idx $target_layer_idx --result_path $result_path --mask_ratio $mask_ratio "

        if [ "$log_output" = true ]; then
            echo "Running target_layer_idx=$target_layer_idx" >> "$log_path"
            echo "Mask ratio=$mask_ratio" >> "$log_path"
            echo "Executing: $cmd" >> "$log_path"
            nohup bash -c "$cmd" >> "$log_path" 2>&1
        else
            echo "Running target_layer_idx=$target_layer_idx"
            echo "Mask ratio=$mask_ratio"
            eval "$cmd"
        fi
        
        sleep 1  # 避免瞬间启动多个任务造成冲突
    done
done
# 本批次执行完后删除