#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
task=$1
model="llava"
method="rel_att"

command="python run.py --model $model --task $task --method $method"
echo "Executing: $command"
eval $command &

wait

# 测试
python utils/get_score.py --data_dir ./results --save_path ./