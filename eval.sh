#!/bin/bash
get_best_gpu() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk '{print $1}' \
    | awk 'BEGIN{max=-1; idx=-1} {if ($1+0 > max) {max=$1; idx=NR-1}} END{print idx}'
}

GPU_ID=$(get_best_gpu)
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU $CUDA_VISIBLE_DEVICES"

task="vqav2"
model="llava_new"
method="rel_att"
max_sample=50000
batch_size=1
# ckpt_path="/ai/teacher/ssz/layer_task/mllms_know/results/ckpts/mvsa_s-woadaloss-0424"
# lora_name="mvsa_s-woadaloss-0424"

result_path="result_$CUDA_VISIBLE_DEVICES.json"
attn_layer_idx=17
# json_file="/ai/teacher/ssz/layer_task/mllms_know/results/jsons/llava-gqa-rel_att.json"

for attn_layer_idx in 17
do
    echo "Running with target_layer_idx=$target_layer_idx"

    lora_name=$task

    command="python _eval.py \
        --model $model \
        --task $task \
        --method $method \
        --max_sample $max_sample \
        --lora_name $lora_name \
        --batch_size $batch_size \
        --attn_layer_idx $attn_layer_idx \
        --result_path $result_path
        "

    echo "Executing: $command"
    eval $command

    # 可选`：每轮之后评估一次
    # score_command="python methods/utils/get_score.py --task $task --lora_name $lora_name"
    # echo "Evaluating score..."
    # eval` $score_command
done