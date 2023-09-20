import lightning.pytorch as pl
import torch
from ..calibration import EvaluateECECallback, EvaluateCalibrationLossCallback
from ..explainability import ARTokenF1ExplainabilityCallback
from lightning.pytorch.loggers import CSVLogger
from .utils import TBLogger


def init_trainer_for_explainability(
        results_dir,
        tokenizer,
        hyperparams,
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
    # new_results_dir = "/".join(results_dir.split("/")[:-1])
    # last_dir = results_dir.split("/")[-1]
    tb_logger = TBLogger(
        save_dir=results_dir,
        name="",
        version=""
    )
    csv_logger = CSVLogger(
        save_dir=results_dir,
        name="",
        version=""
    )
    ar_token_f1_callback = ARTokenF1ExplainabilityCallback(
        tokenizer, 
        add_residuals=True, 
        idx_layer=2
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        max_epochs=hyperparams["num_epochs"],
        gradient_clip_val=hyperparams["max_gradient_norm"],
        logger=[tb_logger, csv_logger],
        callbacks=[
            ar_token_f1_callback,
            evaluate_ece_callback,
            evaluate_psr_callback,
        ],
        val_check_interval=hyperparams["eval_every_n_train_steps"],
        enable_checkpointing=False,
        log_every_n_steps=hyperparams["eval_every_n_train_steps"] // 5
    )
    csv_logger.log_hyperparams(hyperparams)
    return trainer