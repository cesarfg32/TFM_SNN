# src/methods/naive.py
from __future__ import annotations
from torch import nn
import torch

class Naive:
    name = "naive"
    def __init__(self): pass
    def penalty(self) -> torch.Tensor:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), dtype=torch.float32, device=dev)
    def before_task(self, model: nn.Module, train_loader, val_loader): pass
    def after_task(self,  model: nn.Module, train_loader, val_loader): pass
