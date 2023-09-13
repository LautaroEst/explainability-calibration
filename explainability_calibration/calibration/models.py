
import torch
from torch import nn, optim


class AffineModel(nn.Module):

    def __init__(self, num_classes, scale="vector", bias=True):
        super().__init__()
        self.num_classes = num_classes
        if scale == "matrix":
            self.scale = nn.Parameter(torch.eye(n=num_classes, dtype=torch.float))
        elif scale == "vector":
            self.scale = nn.Parameter(torch.ones(num_classes,dtype=torch.float))
        elif scale == "scalar":
            self.scale = nn.Parameter(torch.tensor(1.0))
        elif scale == "none":
            self.scale = torch.tensor(1.0)
        else:
            raise ValueError(f"Scale {scale} not valid.")

        if bias:
            self.bias = nn.Parameter(torch.zeros(1,num_classes))
        else:
            self.bias = torch.zeros(1,num_classes)

    def forward(self, logits):
        if self.scale == "matrix":
            return torch.matmul(logits, self.scale) + self.bias
        return logits * self.scale + self.bias
    

class AffineCalibrationTrainer:

    def __init__(self, num_classes, scale, bias, loss_fn, maxiters=100, lr=1e-4, tolerance=1e-6):
        self.model = AffineModel(num_classes, scale, bias)
        self.loss_fn = loss_fn
        self.maxiters = maxiters
        self.lr = lr
        self.tolerance_change = tolerance

    def fit(self, logits, labels):
        self.model.train()
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=self.lr,
            max_iter=self.maxiters,
            tolerance_change=self.tolerance
        )
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn(logits, labels)
            loss.backward()
            return loss
        optimizer.step(closure)
    
    def calibrate(self,logits):
        self.model.eval()
        with torch.no_grad():
            calibrated_logits = self.model(logits)
        return calibrated_logits        