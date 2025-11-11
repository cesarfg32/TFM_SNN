from __future__ import annotations
import torch
from torch import nn
from .base import BaseMethod

class Naive(BaseMethod):
    name = "naive"
    def __init__(self, *, device=None, loss_fn: nn.Module | None = None):
        super().__init__(device=device, loss_fn=loss_fn)

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def before_task(self, model: nn.Module, train_loader, val_loader): ...
    def after_task(self,  model: nn.Module, train_loader, val_loader): ...
