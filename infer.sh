#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

task="mvsa_s"
model="llava"
method="rel_att"
max_sample=-1

infer_command="python _run.py --model $model --task $task --method $method --max_sample $max_sample"
score_command="python methods/utils/get_score.py --data_dir ./results --save_path ./"

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
