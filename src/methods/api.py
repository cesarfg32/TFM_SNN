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
    def after_task(self,  model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None: ...
    # Hook opcional: algunos métodos (p.ej. Rehearsal) lo exponen
    # No es obligatorio implementarlo en todos.
    # def prepare_train_loader(self, train_loader: DataLoader) -> DataLoader: ...
