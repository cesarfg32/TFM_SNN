# src/methods/composite.py
from __future__ import annotations
from typing import Sequence
from torch import nn
import torch
from .api import ContinualMethod

class CompositeMethod:
    def __init__(self, methods: Sequence[ContinualMethod]):
        assert len(methods) >= 1
        self.methods = list(methods)
        self.name = "+".join(m.name for m in self.methods)

    def penalty(self) -> torch.Tensor:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.zeros((), dtype=torch.float32, device=dev)
        for m in self.methods:
            total = total + m.penalty()
        return total

    def before_task(self, model: nn.Module, train_loader, val_loader):
        for m in self.methods:
            m.before_task(model, train_loader, val_loader)

    def after_task(self, model: nn.Module, train_loader, val_loader):
        for m in self.methods:
            m.after_task(model, train_loader, val_loader)
