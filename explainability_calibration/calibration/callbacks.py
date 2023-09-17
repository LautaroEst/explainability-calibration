import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
import torch
import torch.nn.functional as F
from .losses import LogLoss, Brier
from .models import AffineCalibrationTrainer

class EvaluateECECallback(Callback):

    def __init__(self, n_bins=15):
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.logits_batches = []
        self.labels_batches = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.logits_batches.append(outputs["logits"])
        self.labels_batches.append(batch["label"])

    def on_validation_epoch_end(self, trainer, pl_module):
        # Assuming pl_module.num_labels > 1
        logits = torch.vstack(self.logits_batches)
        labels = torch.hstack(self.labels_batches)
        ece = self._compute_ece(logits, labels)
        pl_module.log("val_ece",ece)
        self.logits_batches.clear()
        self.labels_batches.clear()

    def _compute_ece(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece



class EvaluateCalibrationLossCallback(Callback):
    """
        Compute the calibration loss as the relative difference 
        between the calibrated posteriors and the originals. 
        Calibration is done with the affine calibration method 
        by minimizing a PSR which can be log-loss or brier.
    """

    MODELS = [
        "temperature scaling", # Only one parameter (scaling)
        "vector scaling", # Two parameters (scaling and shift) per class
        "matrix scaling", # K x (K+1) parameters, where K is the number of classes
        "bias only", # One parameter per class (shift)
    ]

    PSRS = [
        "log-loss",
        "normalized log-loss",
        "brier",
        "normalized brier"
    ]

    def __init__(
        self, 
        model="temperature scaling", 
        psr="log-loss", 
        maxiters=100, 
        lr=1e-4, 
        tolearnce=1e-6,
        stratified=False,
        n_folds=5,
        seed=None
    ):
        super().__init__()

        if psr == "log-loss":
            loss_fn = LogLoss(norm=False)
        elif psr == "normalized log-loss":
            loss_fn = LogLoss(norm=True)
        elif psr == "brier":
            loss_fn = Brier(norm=False)
        elif psr == "normalized brier":
            loss_fn = Brier(norm=True)
        else:
            raise ValueError(f"PSR {psr} not supported")

        if model == "matrix scaling":
            scale = "matrix"
            bias = True
        elif model == "vector scaling":
            scale = "vector"
            bias = True
        elif model == "temperature scaling":
            scale = "scalar"
            bias = False
        elif model == "bias only":
            scale = "none"
            bias = True
        else:
            raise ValueError(f"Calibration model {model} not supported")

        self.model = model
        self.psr = psr
        self.scale = scale
        self.bias = bias
        self.loss_fn = loss_fn
        self.maxiters = maxiters
        self.lr = lr
        self.tolearnce = tolearnce
        self.stratified = stratified
        self.n_folds = n_folds
        self.seed = seed
        self.logits_batches = []
        self.labels_batches = []
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.logits_batches.append(outputs["logits"])
        self.labels_batches.append(batch["label"])

    def on_validation_epoch_end(self, trainer, pl_module):
        # Assuming pl_module.num_labels > 1
        logits = torch.vstack(self.logits_batches)
        labels = torch.hstack(self.labels_batches)
        calibrated_logits = self._calibrate_logits(logits, labels)
        calibration_loss = self._compute_calibration_loss(logits, calibrated_logits, labels)
        
        pl_module.log(f"val_cal_loss_{self.psr}_{self.model}",calibration_loss)
        self.logits_batches.clear()
        self.labels_batches.clear()

    def _calibrate_logits(self, logits, labels, condition_ids=None):
        """
            This method performs calibration with cross validation.
        """
        num_classes = logits.size(1)

        if self.stratified:
            if condition_ids is not None:
                skf = StratifiedGroupKFold(n_splits=self.nfolds, shuffle=True, random_state=self.seed)
            else:
                # Use StratifiedKFold in this case for backward compatibility
                skf = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=self.seed)
        else:
            if condition_ids is not None:
                skf = GroupKFold(n_splits=self.nfolds)
            else:
                skf = KFold(n_splits=self.nfolds, shuffle=True, random_state=self.seed)

        calibrated_logits = torch.zeros(logits.shape)
        for trni, tsti in skf.split(logits, labels, condition_ids):
            trainer = AffineCalibrationTrainer(
                num_classes=num_classes,
                scale=self.scale,
                bias=self.bias,
                loss_fn=self.loss_fn,
                maxiters=self.maxiters,
                lr=self.lr,
                tolerance=self.tolerance
            )
            trainer.fit(logits[trni], labels[trni])
            calibrated_logits[tsti] = trainer.calibrate(logits[tsti])

        return calibrated_logits

    def _compute_calibration_loss(self, logits, calibrated_logits, labels):
        raw = self.loss_fn(logits, labels)
        cal = self.loss_fn(calibrated_logits, labels)
        return (raw-cal)/raw*100