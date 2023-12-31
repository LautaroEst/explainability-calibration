#!/bin/bash
#
#$ -S /bin/bash
#$ -N explainability_over_time2
#$ -o /homes/eva/q/qestienne/projects/explainability/logs/explainability_over_time2_out.log
#$ -e /homes/eva/q/qestienne/projects/explainability/logs/explainability_over_time2_err.log
#$ -q long.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=16G,ram_free=64G,mem_free=30G
#

# Configure environment
project_name="explainability"
env_name="fair"
source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project_name
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)
script_name="explainability_over_time"



declare -a datasets=(
    # "dynasent"
    # "cose"
    "sst2"
)

declare -a models=(
    # "albert-base-v2"
    # "albert-large-v2"
    "nreimers--MiniLM-L6-H384-uncased"
    # "microsoft--MiniLM-L12-H384-uncased"
    # "albert-xlarge-v2"
    "distilbert-base-uncased"
    "distilroberta-base"
    "bert-base-uncased"
    "roberta-base"
    # "facebook--muppet-roberta-base"
    # "microsoft--deberta-v3-base"
    # "albert-xxlarge-v2"
    # "bert-large-uncased"
    # "roberta-large"
    # "facebook--muppet-roberta-large"
    # "microsoft--deberta-v3-large"
)

# Run model selection
seed=23840
for dataset in "${datasets[@]}"; do
    for base_model in "${models[@]}"; do
        echo ">>> Running ${script_name} for ${base_model} on ${dataset}..."
        mkdir -p results/${script_name}/${dataset}/${base_model}
        python scripts/python/explainability_over_time.py \
            --root_directory=. \
            --base_model=${base_model} \
            --dataset=$dataset \
            --seed=$seed \
            --hyperparameters_config=./configs/${script_name}/${base_model}_${dataset}.json
    done
done

# Deactivate environment
conda deactivate