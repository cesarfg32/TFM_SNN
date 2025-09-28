# src/methods/as_snn.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from torch.utils.data import DataLoader
from torch import nn
import torch

class AS_SNN:
    name = "as-snn"
    def __init__(self, alpha: float = 1e-3, target_rate: float = 0.05, **kw):
        self.alpha = float(alpha)
        self.target_rate = float(target_rate)
        # TODO: buffers de actividad por capa

    def penalty(self) -> torch.Tensor:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), device=dev)

    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        pass

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # TODO: aplicar escalado sináptico hacia self.target_rate
        pass
