# src/methods/sa_snn.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from torch.utils.data import DataLoader
from torch import nn
import torch

class SA_SNN:
    name = "sa-snn"
    def __init__(self, sparsity: float = 0.2, gate_temp: float = 1.0, **kw):
        self.sparsity = float(sparsity)
        self.gate_temp = float(gate_temp)
        # TODO: inicializar máscaras/gates por capa

    def penalty(self) -> torch.Tensor:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), device=dev)

    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # TODO: preparar subpoblaciones/máscaras activas por tarea
        pass

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # TODO: consolidar selección si aplica
        pass
