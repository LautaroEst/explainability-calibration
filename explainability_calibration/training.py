import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from .calibration import EvaluateECECallback, EvaluatePSRCalibrationCallback

def init_trainer_with_callbacks(results_dir, save_every_n_train_steps, num_epochs, max_gradient_norm):
    # global_step_callback = ModelCheckpoint(
    #     dirpath=results_dir,
    #     filename="checkpoint-{step}",
    #     monitor="step",
    #     mode="max",
    #     save_top_k=-1,
    #     every_n_train_steps=save_every_n_train_steps
    # )
    evaluate_calibration_callback = EvaluateECECallback(n_bins=10)
    evaluate_explainability_calback = None
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        max_epochs=num_epochs,
        gradient_clip_val=max_gradient_norm,
        default_root_dir=results_dir,
        callbacks=[evaluate_calibration_callback],
        val_check_interval=save_every_n_train_steps
    )
    return trainer