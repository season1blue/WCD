#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

task="mvsa_s"
model="llava"
method="rel_att"
max_sample=-1
ckpt_path="/ai/teacher/ssz/layer_task/mllms_know/results/ckpts/mvsa_s-woadaloss-0424"
# json_file="/ai/teacher/ssz/layer_task/mllms_know/results/jsons/llava-gqa-rel_att.json"

infer_command="python _run.py --model $model --task $task --method $method --max_sample $max_sample --ckpt_path $ckpt_path"
score_command="python methods/utils/get_score.py --ckpt_path $ckpt_path"

if [ "$1" == "infer" ]; then
    echo "Executing: $infer_command"
    eval $infer_command

elif [ "$1" == "score" ]; then
    echo "Executing: $score_command"
    eval $score_command

elif [ "$1" == "all" ]; then
    echo "Executing: $infer_command"
    eval $infer_command
    wait
    echo "Executing: $score_command"
    eval $score_command

else
    echo "Usage: bash main.sh [infer|score|all]"
    exit 1
fi
