# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence
from torch import nn
import torch

from .base import BaseMethod


class CompositeMethod(BaseMethod):
    """
    Composite de métodos de aprendizaje continuo.
    - Suma penalizaciones.
    - Propaga hooks.
    - Encadena prepare_train_loader() si existe.
    - Garantiza que EWC se ejecute el ÚLTIMO en after_task (para snapshot de θ* correcto).
    """
    def __init__(self, methods: Sequence[BaseMethod]):
        super().__init__(device=(methods[0].device if methods else None))
        assert len(methods) >= 1, "CompositeMethod requiere al menos un método"
        self.methods = list(methods)
        self.name = "+".join(m.name for m in self.methods)

    def penalty(self) -> torch.Tensor:
        total: torch.Tensor | None = None
        for m in self.methods:
            p = m.penalty()
            if total is None:
                total = p
            else:
                if hasattr(p, "device") and hasattr(total, "device") and (p.device != total.device):
                    p = p.to(total.device)
                total = total + p
        if total is None:
            total = torch.zeros((), device=self.device, dtype=torch.float32)
        return total

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        for m in self.methods:
            m.before_task(model, train_loader, val_loader)

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        # 1ª pasada: todos MENOS EWC
        for m in self.methods:
            if str(getattr(m, "name", "")).lower() == "ewc":
                continue
            m.after_task(model, train_loader, val_loader)
        # 2ª pasada: EWC al final
        for m in self.methods:
            if str(getattr(m, "name", "")).lower() == "ewc":
                m.after_task(model, train_loader, val_loader)

    def before_epoch(self, model: nn.Module, epoch: int) -> None:
        for m in self.methods:
            m.before_epoch(model, epoch)

    def after_epoch(self, model: nn.Module, epoch: int) -> None:
        for m in self.methods:
            m.after_epoch(model, epoch)

    def before_batch(self, model: nn.Module, batch) -> None:
        for m in self.methods:
            m.before_batch(model, batch)

    def after_batch(self, model: nn.Module, batch, loss) -> None:
        for m in self.methods:
            m.after_batch(model, batch, loss)

    def prepare_train_loader(self, train_loader):
        loader = train_loader
        for m in self.methods:
            if hasattr(m, "prepare_train_loader"):
                loader = m.prepare_train_loader(loader) # type: ignore[attr-defined]
        return loader