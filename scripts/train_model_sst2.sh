#!/bin/bash
#
#$ -S /bin/bash
#$ -N training_sst2
#$ -o /homes/eva/q/qestienne/projects/interpretability/training_sst2_out.log
#$ -e /homes/eva/q/qestienne/projects/interpretability/training_sst2_err.log
#$ -q all.q
#$ -l matylda3=0.5,gpu=1,gpu_ram=16G,ram_free=64G,mem_free=10G
#

project="interpretability"
env_name="fair"

source ~/.bashrc
conda activate $env_name
cd $PROJECTS_DIR/$project
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1)

DATASET="sst2"

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


for BASE_MODEL in "${models[@]}"
do
  echo "Training "${BASE_MODEL}...
  mkdir -p ${PROJECTS_DIR}/${project}/results/${DATASET}/${BASE_MODEL}
  python scripts/python/train_model.py \
    --root_directory=${PROJECTS_DIR}/${project} \
    --base_model=${BASE_MODEL} \
    --dataset=${DATASET} \
    --n_labels=1 \
    --store_model_with_best="val_acc" \
    --eval_every_epoch=1 \
    --batch_size=16 \
    --num_epochs=3 \
    --learning_rate=0.00002 \
    --weight_decay=0.01 \
    --warmup_proportion 0.1
done

conda deactivate