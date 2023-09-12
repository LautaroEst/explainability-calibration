import torch

def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    preds = torch.max(logits, axis=1).indices.type(torch.long) if len(logits.shape) > 1 else (logits > 0).type(torch.long).view(-1)
    targets = target.view(-1).type(torch.long)
    acc = torch.mean((preds == targets).type(torch.float))
    return acc