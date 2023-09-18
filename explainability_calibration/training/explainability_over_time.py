import lightning.pytorch as pl
import torch
from ..calibration import EvaluateECECallback, EvaluateCalibrationLossCallback
from ..explainability import ARTokenF1ExplainabilityCallback


def init_trainer_for_explainability(
        results_dir,
        tokenizer,
        hyperparameters,
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
    ar_token_f1_callback = ARTokenF1ExplainabilityCallback(
        tokenizer, 
        add_residuals=True, 
        idx_layer=2
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        max_epochs=hyperparameters["num_epochs"],
        gradient_clip_val=hyperparameters["max_gradient_norm"],
        default_root_dir=results_dir,
        callbacks=[
            # evaluate_ece_callback,
            # evaluate_psr_callback,
            ar_token_f1_callback
        ],
        val_check_interval=hyperparameters["eval_every_n_train_steps"] // 5
    )
    return trainer