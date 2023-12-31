
import argparse
import glob
import json
import random

import torch
import explainability_calibration as ec
import numpy as np
import os
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Args config for training script."
    )
    parser.add_argument("--root_directory", type=str, required=True, help="Root directory of the proyect.")
    parser.add_argument("--base_model", type=str, required=True, help="Tranformers model name")
    parser.add_argument("--dataset", type=ec.data.SupportedDatasets, required=True, choices=list(ec.data.SupportedDatasets))
    parser.add_argument("--seed", type=int, default=23840, help="Random seed for reproducibility.")
    parser.add_argument("--hyperparameters_config", type=str, help="Config file with hyperparameters values.")
    args = parser.parse_args()

    with open(args.hyperparameters_config, "r") as f:
        hyperparameters = json.load(f)
    setattr(args,"hyperparameters",hyperparameters)

    return args


def main():

    # Parse arguments and initialize utilities:
    args = parse_args()
    
    # Random state
    random.seed(args.seed)     
    np.random.seed(args.seed)  
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load the data in train_on_non_annotated mode:
    print(f"Loading {args.dataset} dataset...")
    datadict = ec.data.load_dataset(
        args.dataset,
        args.root_directory,
        mode="trainon_nonannot_valon_nonannot_teston_annot"
    )

    # Load the pre-trained tokenizer:
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model.replace("--","/"), 
        do_lower_case="uncased" in args.base_model, 
        local_files_only=True
    )

    # Create the train and validation dataloaders:
    print(f"Creating dataloaders...")
    train_loader = ec.data.LoaderWithDynamicPadding(
        dataset=datadict["train"],
        tokenizer=tokenizer,
        batch_size=args.hyperparameters["batch_size"],
        shuffle=True,
        random_state=args.seed
    )
    test_loader = ec.data.LoaderWithDynamicPadding(
        dataset=datadict["test"],
        tokenizer=tokenizer,
        batch_size=args.hyperparameters["batch_size"],
        shuffle=False,
        random_state=None
    )

    # Load the pre-trained model to perform Sequence Classification:
    print(f"Loading {args.base_model} pre-trained model...")
    model = ec.modeling.SequenceClassificationModel(
        base_model=args.base_model,
        num_labels=datadict.num_labels,
        learning_rate=args.hyperparameters["learning_rate"],
        weight_decay=args.hyperparameters["weight_decay"],
        warmup_proportion=args.hyperparameters["warmup_proportion"],
        num_train_optimization_steps=args.hyperparameters["num_epochs"] * len(train_loader),
    )

    # Results directory
    results_dir = os.path.join(args.root_directory,"results/explainability_over_time",args.dataset,args.base_model)

    # Init trainer:
    trainer = ec.training.init_trainer_for_explainability(
        results_dir, 
        tokenizer,
        args.hyperparameters,
        args.seed
    )

    # Train, validate and save checkpoints:
    possible_ckpts = glob.glob(os.path.join(results_dir,f"{args.seed}/*.ckpt"))
    ckpt_path = possible_ckpts[0] if len(possible_ckpts) > 0 else None
    trainer.fit(model, train_loader, test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()    
