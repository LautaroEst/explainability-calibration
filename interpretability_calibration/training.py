import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

def init_trainer(results_dir, store_model_with_best, num_epochs, max_gradient_norm):
    mode = "min" if store_model_with_best in ["mse", "l1"] else "max"
    best_model_callback = ModelCheckpoint(
        dirpath=results_dir,
        filename="checkpoint-best",
        monitor=store_model_with_best,
        mode=mode,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    epoch_callback = ModelCheckpoint(
        dirpath=results_dir,
        filename="checkpoint-{epoch:01d}",
        monitor="epoch",
        mode="max",
        every_n_epochs=1,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        max_epochs=num_epochs,
        gradient_clip_val=max_gradient_norm,
        default_root_dir=results_dir,
        callbacks=[best_model_callback, epoch_callback]
    )
    return trainer