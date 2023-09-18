
import argparse
import json
import explainability_calibration as ec
import numpy as np
import os
from transformers import AutoTokenizer


MAX_INT_GENERATOR = 1000


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
        hyperparameters_dicts = [json.loads(hyperparameters) for hyperparameters in f.read().splitlines()]
    setattr(args,"hyperparameters_dicts",hyperparameters_dicts)
    return args


def main():

    # Parse arguments and initialize utilities:
    args = parse_args()
    
    # Random state
    rs = np.random.RandomState(args.seed)

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
        args.base_model, 
        do_lower_case="uncased" in args.base_model, 
        local_files_only=True
    )

    for hyperparams_id, hyperparams in enumerate(args.hyperparameters_dicts):

        # Results directory
        results_dir = os.path.join(args.root_directory,"results/model_selection",args.dataset,args.base_model)

        # Create the train and validation dataloaders:
        train_loader = ec.data.LoaderWithDynamicPadding(
            dataset=datadict["train"],
            tokenizer=tokenizer,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            random_state=rs.randint(0,MAX_INT_GENERATOR)
        )
        validation_loader = ec.data.LoaderWithDynamicPadding(
            dataset=datadict["validation"],
            tokenizer=tokenizer,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
            random_state=None
        )

        # Instantiate model:
        model = ec.modeling.SequenceClassificationModel(
            base_model=args.base_model,
            num_labels=datadict.num_labels,
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            warmup_proportion=hyperparams["warmup_proportion"],
            num_train_optimization_steps=hyperparams["num_epochs"] * len(train_loader)
        )

        # Init trainer:
        trainer = ec.training.init_trainer_for_model_selection(
            results_dir,
            hyperparams_id,
            hyperparams,
            rs.randint(0,MAX_INT_GENERATOR)
        )

        # Train, validate and save checkpoints:
        trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    main()    
