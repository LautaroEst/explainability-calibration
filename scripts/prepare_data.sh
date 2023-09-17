#!/bin/bash

project="interpretability"
env_name="fair"

source ~/.bashrc
conda activate $env_name

# Make link to data
if [ ! -d data ]; then
    mkdir data
    ln -s $DATA_DIR/fair-data data/fair-data
    ln -s $DATA_DIR/sst2 data/sst2
    ln -s $DATA_DIR/dynasent-v1.1 data/dynasent-v1.1
    ln -s $DATA_DIR/cose data/cose
fi
conda deactivate
