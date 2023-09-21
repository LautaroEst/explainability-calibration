import os
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from .utils import TBLogger
import torch
from ..calibration import EvaluateECECallback, EvaluateCalibrationLossCallback

def init_trainer_for_model_selection(
        results_dir,
        hyperparams,
        hyperparams_id,
        random_state
    ):
    evaluate_ece_callback = EvaluateECECallback(n_bins=10)
    evaluate_psr_callback = EvaluateCalibrationLossCallback(
        model="vector scaling",
        psr="log-loss",
        maxiters=100,
        lr=1e-4,
        tolerance=1e-6,
        stratified=False,
        n_folds=5,
        random_state=random_state
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir,f"hparams_{hyperparams_id}/{random_state}"),
        filename="checkpoint-{global_step}",
        monitor="step",
        mode="max",
        every_n_train_steps=hyperparams["eval_every_n_train_steps"],
    )

    tb_logger = TBLogger(
        save_dir=results_dir,
        name="",
        version=f"hparams_{hyperparams_id}/{random_state}"
    )
    csv_logger = CSVLogger(
        save_dir=results_dir,
        name=f"hparams_{hyperparams_id}/{random_state}"
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        max_epochs=hyperparams["num_epochs"],
        gradient_clip_val=hyperparams["max_gradient_norm"],
        logger=[tb_logger, csv_logger],
        callbacks=[
            evaluate_ece_callback,
            evaluate_psr_callback,
            checkpoint_callback
        ],
        val_check_interval=hyperparams["eval_every_n_train_steps"],
        enable_checkpointing=True,
        log_every_n_steps=hyperparams["eval_every_n_train_steps"] // 5
    )
    csv_logger.log_hyperparams(hyperparams)
    return trainer