#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../../
# 替换下面的 JSON 文件名为你实际生成的文件名
export LOG_FILE="/gemini/space/evol/sunminghao/GCG/experiments/results/individual_behaviors_llama2_gcg_offset0_20260113-20:27:44.json"

python -u ../evaluate_individual.py \
    --config="../configs/individual_llama2.py" \
    --config.train_data="../../data/advbench/harmful_behaviors.csv" \
    --config.logfile="${LOG_FILE}" \
    --config.n_train_data=10 \
    --config.data_offset=0 \
    --config.n_test_data=0