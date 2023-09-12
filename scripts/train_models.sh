#!/bin/bash
#
#$ -S /bin/bash
#$ -N train_models
#$ -o /homes/eva/q/qestienne/projects/interpretability/logs/train_models_out.log
#$ -e /homes/eva/q/qestienne/projects/interpretability/logs/train_models_err.log
#$ -q all.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=16G,ram_free=64G,mem_free=10G
#

project="interpretability"
env_name="fair"

source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)

SEED=23840

## declare an array variable
declare -a models=(
# "albert-base-v2"
# "albert-large-v2"
# "nreimers/MiniLM-L6-H384-uncased"
# "microsoft/MiniLM-L12-H384-uncased"
# "albert-xlarge-v2"
# "distilbert-base-uncased"
"distilroberta-base"
# "bert-base-uncased"
# "roberta-base"
# "facebook/muppet-roberta-base"
# "microsoft/deberta-v3-base"
# "albert-xxlarge-v2"
# "bert-large-uncased"
# "roberta-large"
# "facebook/muppet-roberta-large"
# "microsoft/deberta-v3-large"
)

# DATASET="sst2"
# for BASE_MODEL in "${models[@]}"
# do
#   echo "Training "${BASE_MODEL} on ${DATASET} dataset...
#   mkdir -p ${PROJECTS_DIR}/${project}/results/${DATASET}/${BASE_MODEL}
#   python scripts/python/train_model.py \
#     --root_directory=${PROJECTS_DIR}/${project} \
#     --base_model=${BASE_MODEL} \
#     --dataset=$DATASET \
#     --n_labels=2 \
#     --store_model_with_best="val_acc" \
#     --eval_every_epoch=1 \
#     --batch_size=32 \
#     --num_epochs=3 \
#     --learning_rate=0.00002 \
#     --weight_decay=0.01 \
#     --warmup_proportion 0.1 \
#     --max_gradient_norm 10.0 \
#     --seed $SEED
# done

# DATASET="dynasent"
# for BASE_MODEL in "${models[@]}"
# do
#   echo "Training "${BASE_MODEL} on ${DATASET} dataset...
#   mkdir -p ${PROJECTS_DIR}/${project}/results/${DATASET}/${BASE_MODEL}
#   python scripts/python/train_model.py \
#     --root_directory=${PROJECTS_DIR}/${project} \
#     --base_model=${BASE_MODEL} \
#     --dataset=$DATASET \
#     --n_labels=3 \
#     --store_model_with_best="val_acc" \
#     --eval_every_epoch=1 \
#     --batch_size=32 \
#     --num_epochs=3 \
#     --learning_rate=0.00002 \
#     --weight_decay=0.01 \
#     --warmup_proportion 0.1 \
#     --max_gradient_norm 10.0 \
#     --seed $SEED
# done

DATASET="cose"
for BASE_MODEL in "${models[@]}"
do
  echo "Training "${BASE_MODEL} on ${DATASET} dataset...
  mkdir -p ${PROJECTS_DIR}/${project}/results/${DATASET}/${BASE_MODEL}
  python scripts/python/train_model.py \
    --root_directory=${PROJECTS_DIR}/${project} \
    --base_model=${BASE_MODEL} \
    --dataset=$DATASET \
    --n_labels=5 \
    --store_model_with_best="val_acc" \
    --eval_every_epoch=1 \
    --batch_size=16 \
    --num_epochs=3 \
    --learning_rate=0.00001 \
    --weight_decay=0.01 \
    --warmup_proportion 0.0 \
    --max_gradient_norm 10.0 \
    --seed $SEED
done

DATASET="cose_simplified"
for BASE_MODEL in "${models[@]}"
do
  echo "Training "${BASE_MODEL} on ${DATASET} dataset...
  mkdir -p ${PROJECTS_DIR}/${project}/results/${DATASET}/${BASE_MODEL}
  python scripts/python/train_model.py \
    --root_directory=${PROJECTS_DIR}/${project} \
    --base_model=${BASE_MODEL} \
    --dataset=$DATASET \
    --n_labels=2 \
    --store_model_with_best="val_acc" \
    --eval_every_epoch=1 \
    --batch_size=16 \
    --num_epochs=3 \
    --learning_rate=0.00001 \
    --weight_decay=0.01 \
    --warmup_proportion 0.0 \
    --max_gradient_norm 10.0 \
    --seed $SEED
done

conda deactivate