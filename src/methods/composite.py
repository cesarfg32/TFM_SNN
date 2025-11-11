# src/methods/composite.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Sequence
from torch import nn
import torch
from .api import ContinualMethod

class CompositeMethod(ContinualMethod):
    """
    Composite de métodos de aprendizaje continuo.
    - Suma las penalizaciones de cada submétodo.
    - Propaga before/after hooks a todos.
    - Encadena prepare_train_loader si existe.
    """
    def __init__(self, methods: Sequence[ContinualMethod]):
        assert len(methods) >= 1, "CompositeMethod requiere al menos un método"
        self.methods = list(methods)
        self.name = "+".join(m.name for m in self.methods)

    def penalty(self) -> torch.Tensor:
        total: torch.Tensor | None = None
        for m in self.methods:
            p = m.penalty()
            total = p if total is None else (total + p.to(total.device))
        return total  # type: ignore[return-value]

    def prepare_train_loader(self, train_loader):
        loader = train_loader
        for m in self.methods:
            if hasattr(m, "prepare_train_loader"):
                loader = m.prepare_train_loader(loader)  # type: ignore[attr-defined]
        return loader

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        for m in self.methods:
            m.before_task(model, train_loader, val_loader)

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        for m in self.methods:
            m.after_task(model, train_loader, val_loader)
