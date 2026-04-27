#!/bin/bash

# 设置环境变量
export WANDB_MODE=disabled
export PYTHONPATH=$PYTHONPATH:/gemini/space/evol/sunminghao/GCG

# 参数设置
export model="llama2"       # 选项: llama2, vicuna
export setup="behaviors"    # 选项: behaviors, strings
export batch_size=128       # 根据显存调整
export n_steps=100          # 攻击步数

# 创建结果文件夹
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

# 循环运行: 
# seq 0 100 500 会生成: 0, 100, 200, 300, 400, 500
# 这样每次偏移 100 条数据，直到跑完
for data_offset in $(seq 0 100 500)
do
    echo "Running attack with offset: $data_offset"

    python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/individual_${setup}_${model}_gcg_offset${data_offset}" \
        --config.n_train_data=100 \
        --config.data_offset=$data_offset \
        --config.n_steps=$n_steps \
        --config.test_steps=20 \
        --config.batch_size=$batch_size

done