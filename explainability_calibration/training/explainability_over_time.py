import lightning.pytorch as pl
import torch
from ..calibration import EvaluateECECallback, EvaluateCalibrationLossCallback
from ..explainability import EvaluateExplainabilityCallback


def init_trainer_for_explainability(
        results_dir, 
        save_every_n_train_steps, 
        num_epochs, 
        max_gradient_norm,
        random_state=None
    ):
    evaluate_ece_callback = EvaluateECECallback(n_bins=10)
    evaluate_psr_callback = EvaluateCalibrationLossCallback(
        model="vector scaling",
        psr="log-loss",
        maxiters=100,
        lr=1e-4,
        tolearnce=1e-6,
        stratified=False,
        n_folds=5,
        seed=random_state
    )
    evaluate_explainability_calback = EvaluateExplainabilityCallback(
        
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        max_epochs=num_epochs,
        gradient_clip_val=max_gradient_norm,
        default_root_dir=results_dir,
        callbacks=[
            evaluate_ece_callback,
            evaluate_psr_callback,
            evaluate_explainability_calback
        ],
        val_check_interval=save_every_n_train_steps
    )
    return trainer