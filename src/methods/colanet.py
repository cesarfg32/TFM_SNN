# src/methods/colanet.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from torch.utils.data import DataLoader
from torch import nn
import torch

class CoLaNET:
    name = "colanet"
    def __init__(self, columns: int = 4, col_reg: float = 1e-4, **kw):
        self.columns = int(columns)
        self.col_reg = float(col_reg)
        # TODO: asignación/activación de columna por tarea

    def penalty(self) -> torch.Tensor:
        # TODO: si añades regularización inter-columnas, suma aquí
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), device=dev)

    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # TODO: fijar columna activa
        pass

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        pass
