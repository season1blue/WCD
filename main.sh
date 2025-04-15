#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
task="gqa"
model="llava"
method="rel_att"

command="python run.py --model $model --task $task --method $method"
echo "Executing: $command"
eval $command &

wait