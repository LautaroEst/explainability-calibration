import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.utilities import rank_zero_only
import torch
from ..calibration import EvaluateECECallback, EvaluateCalibrationLossCallback

class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        metrics.pop('global_step', None)
        return super().log_metrics(metrics, step)


def init_trainer_for_model_selection(
        results_dir,
        grid_num,
        hyperparams,
        random_state=None
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
        seed=random_state
    )
    tb_logger = TBLogger(
        save_dir=results_dir,
        name="",
        version=f"hparams_{grid_num:02d}"
    )
    csv_logger = CSVLogger(
        save_dir=results_dir,
        name="",
        version=f"hparams_{grid_num:02d}"
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
        ],
        val_check_interval=hyperparams["eval_every_n_train_steps"],
        enable_checkpointing=False,
        log_every_n_steps=hyperparams["eval_every_n_train_steps"] // 5
    )
    csv_logger.log_hyperparams(hyperparams)
    return trainer