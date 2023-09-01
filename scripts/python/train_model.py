
import interpretability_calibration as ic
from transformers import AutoModelForSequenceClassification
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Args config for training script."
    )
    parser.add_argument("--root_directory", type=str, required=True, help="Root directory of the proyect.")
    parser.add_argument("--base_model", type=str, required=True, help="Tranformers model name")
    parser.add_argument("--dataset", type=ic.SupportedDatasets, required=True, choices=list(ic.SupportedDatasets))
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
    args = parse_args()    
    data = ic.data.load_dataset(args.dataset,args.root_directory)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, local_files_only=True)
    


if __name__ == "__main__":
    main()