import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

def init_trainer(results_dir, save_every_n_train_steps, num_epochs, max_gradient_norm):
    ## TODO: Replace this callback with the new ones.
    global_step_callback = ModelCheckpoint(
        dirpath=results_dir,
        filename="checkpoint-{step}",
        monitor="step",
        mode="max",
        save_top_k=-1,
        every_n_train_steps=save_every_n_train_steps
    )
    ## TODO: code the callbacks
    save_predictions_callback = None
    evaluate_calibration_callback = None
    evaluate_explainability_calback = None
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        max_epochs=num_epochs,
        gradient_clip_val=max_gradient_norm,
        default_root_dir=results_dir,
        callbacks=[global_step_callback],
        val_check_interval=save_every_n_train_steps
    )
    return trainer