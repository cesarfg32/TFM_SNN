# src/methods/api.py
from __future__ import annotations
from typing import Protocol
from torch.utils.data import DataLoader
from torch import nn
import torch

class ContinualMethod(Protocol):
    name: str
    def penalty(self) -> torch.Tensor: ...
    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None: ...
    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None: ...
