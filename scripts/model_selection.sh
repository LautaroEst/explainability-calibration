#!/bin/bash
#
#$ -S /bin/bash
#$ -N model_selection
#$ -o /homes/eva/q/qestienne/projects/interpretability/logs/model_selection_out.log
#$ -e /homes/eva/q/qestienne/projects/interpretability/logs/model_selection_err.log
#$ -q all.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=16G,ram_free=64G,mem_free=10G
#

# Define variables
project_name="explainability"
env_name="fair"
seed=23840
base_model="distilroberta-base"
dataset="sst2"

# Configure environment
source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project_name
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)

# Run model selection
script_name=$(basename $0 .sh)
echo "Training on ${dataset} dataset..."
mkdir -p ./results/${script_name}/${base_model}/${dataset}
python scripts/python/hyperparameter_random_search.py \
  --root_directory=. \
  --base_model=${base_model} \
  --dataset=$dataset \
  --seed=$seed \
  --hyperparameters_config=./configs/${script_name}/${base_model}_${dataset}.json

for s in 0 1 2 3; do
  mkdir -p ./results/${script_name}/${base_model}/${dataset}/seed_${s}
  python scripts/python/compute_model_selection_results.py \
    --root_directory=. \
    --base_model=${base_model} \
    --dataset=$dataset \
    --seed=$s \
    --hyperparameters_config=./results/${script_name}/${base_model}/${dataset}/best_hyperparameters.json
done
# Deactivate environment
conda deactivate