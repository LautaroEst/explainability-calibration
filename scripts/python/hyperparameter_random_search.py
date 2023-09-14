
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
        hyperparameters = json.load(f)
    for name, value in hyperparameters.items():
        setattr(args,name,value)

    return args


def main():

    # Parse arguments and initialize utilities:
    args = parse_args()
    
    # Random state
    rs = np.random.RandomState(args.seed)
    print(f"Random State generator: {rs}")

    # Results directory
    results_dir = os.path.join(args.root_directory,"results/model_selection",args.base_model,args.dataset,str(args.seed))
    print(f"Results will be saved to:\n{results_dir}")

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

    
    # TODO: define grid
    grid = define_grid()
    for hyperparams in grid:

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

        num_train_optimization_steps = int(len(train_loader) / hyperparams["batch_size"]) * hyperparams["num_epochs"]
        model = ec.modeling.SequenceClassificationModel(
            base_model=hyperparams["base_model"],
            num_labels=datadict.num_labels,
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
            warmup_proportion=hyperparams["warmup_proportion"],
            num_train_optimization_steps=num_train_optimization_steps
        )

        # Init trainer:
        trainer = ec.training.init_trainer_for_model_selection(
            results_dir, 
            hyperparams["eval_every_n_train_steps"], 
            hyperparams["num_epochs"], 
            hyperparams["max_gradient_norm"],
            rs.randint(0,MAX_INT_GENERATOR)
        )

    # Train, validate and save checkpoints:
    print("***** Running training *****")
    print("  Num examples = %d", len(datadict['train']))
    print("  Batch size = %d", hyperparams["batch_size"])
    print("  Num steps = %d", len(train_loader)*hyperparams["num_epochs"])
    trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    main()    
