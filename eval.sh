#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

task="mvsa_s"
model="llava"
method="rel_att"
max_sample=-1
batch_size=32
# ckpt_path="/ai/teacher/ssz/layer_task/mllms_know/results/ckpts/mvsa_s-woadaloss-0424"
lora_name="mvsa_s-woadaloss-0424"
# lora_name="NONE"
# json_file="/ai/teacher/ssz/layer_task/mllms_know/results/jsons/llava-gqa-rel_att.json"

command="python _eval.py --model $model --task $task --method $method --max_sample $max_sample --lora_name $lora_name --batch_size $batch_size"
score_command="python methods/utils/get_score.py --task $task --lora_name $lora_name"

echo "Executing: $command"
eval $command
# eval $score_command