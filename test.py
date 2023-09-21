import torch
import lightning.pytorch as pl

def main():
    # with open("results/explainability_over_time/dynasent/distilroberta-base/23840/last.ckpt", "rb") as f:
    #     checkpoint = torch.load(f)
    # print(checkpoint.keys())
    trainer = pl.Trainer(resume_from_checkpoint="results/explainability_over_time/dynasent/distilroberta-base/23840/last.ckpt")
    print(trainer)


if __name__ == "__main__":
    main()