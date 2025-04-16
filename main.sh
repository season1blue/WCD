#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
task=$1
model="llava"
method="rel_att"
max_sample=100

command="python run.py --model $model --task $task --method $method --max_sample $max_sample"
echo "Executing: $command"
eval $command

wait

# 测试
# python utils/get_score.py --data_dir ./results --save_path ./