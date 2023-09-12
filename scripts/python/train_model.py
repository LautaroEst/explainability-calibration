
import argparse
import interpretability_calibration as ic
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
    parser.add_argument("--dataset", type=ic.data.SupportedDatasets, required=True, choices=list(ic.data.SupportedDatasets))
    parser.add_argument("--n_labels", type=int, default=2, help="Number of classes")
    parser.add_argument("--store_model_with_best", type=str, default=None, help="It should be set to the name of an evaluation metric.\nIf set the checkpoint with the best such evaluation\nmetric will be in the 'best' folder.")
    parser.add_argument("--eval_every_epoch", type=int, default=1, help="Evaluation interval in training epochs.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-2, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Train and evaluation batch size.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--max_gradient_norm", type=float, default=10.0, help="Max. norm for gradient norm clipping")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.\nE.g., 0.1 = 10%% of training.")
    parser.add_argument("--seed", type=int, default=23840, help="Random seed for reproducibility.")
    args = parser.parse_args()

    return args


def main():

    # Parse arguments and initialize utilities:
    args = parse_args()
    
    # Random state
    rs = np.random.RandomState(args.seed)
    print(f"Random State generator: {rs}")

    # Results directory
    results_dir = os.path.join(args.root_directory,"results",args.dataset,args.base_model)
    print(f"Results will be saved to:\n{results_dir}")

    # Load the data in train_on_non_annotated mode:
    print(f"Loading {args.dataset} dataset...")
    datadict = ic.data.load_dataset(
        args.dataset,
        args.root_directory,
        mode="train_on_non_annotated"
    )

    # Load the pre-trained tokenizer:
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, 
        do_lower_case="uncased" in args.base_model, 
        local_files_only=True
    )

    # Create the train and validation dataloaders:
    print(f"Creating dataloaders...")
    train_loader = ic.data.LoaderWithDynamicPadding(
        dataset=datadict["train"],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        random_state=rs.randint(0,MAX_INT_GENERATOR)
    )
    validation_loader = ic.data.LoaderWithDynamicPadding(
        dataset=datadict["validation"],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        random_state=None
    )

    # Load the pre-trained model to perform Sequence Classification:
    print(f"Loading {args.base_model} pre-trained model...")
    num_train_optimization_steps = int(len(train_loader) / args.batch_size) * args.num_epochs
    model = ic.modeling.SequenceClassificationModel(
        base_model=args.base_model,
        num_labels=args.n_labels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_proportion=args.warmup_proportion,
        num_train_optimization_steps=num_train_optimization_steps
    )

    # Init trainer:
    trainer = ic.modeling.init_trainer(results_dir, args.store_model_with_best, args.num_epochs, args.max_gradient_norm)

    # Train, validate and save checkpoints:
    print("***** Running training *****")
    print("  Num examples = %d", len(datadict['train']))
    print("  Batch size = %d", args.batch_size)
    print("  Num steps = %d", len(train_loader)*args.num_epochs)
    trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    main()    