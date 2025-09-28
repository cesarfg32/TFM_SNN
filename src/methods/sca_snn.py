# src/methods/sca_snn.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from torch.utils.data import DataLoader
from torch import nn
import torch

class SCA_SNN:
    name = "sca-snn"
    def __init__(self, sim_threshold: float = 0.7, expand_limit: float = 0.25, **kw):
        self.sim_threshold = float(sim_threshold)
        self.expand_limit = float(expand_limit)
        # TODO: métrica de similitud y política de reutilizar/expandir

    def penalty(self) -> torch.Tensor:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), device=dev)

    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # TODO: decidir reutilización vs expansión
        pass

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        pass
