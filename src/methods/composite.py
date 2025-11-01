# src/methods/composite.py
from __future__ import annotations
from typing import Sequence
from torch import nn
import torch

from .api import ContinualMethod


class CompositeMethod(ContinualMethod):
    """
    Composite de métodos de aprendizaje continuo.

    - Se comporta como cualquier ContinualMethod (misma interfaz).
    - Suma las penalizaciones de cada submétodo.
    - Propaga before_task/after_task a todos.
    - Si algún submétodo define `prepare_train_loader(loader)`, se encadena en orden.
      (Este hook es opcional; *no* forma parte del protocolo actual).
    """

    def __init__(self, methods: Sequence[ContinualMethod]):
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
                # Alinear dispositivos para evitar mismatches CPU/GPU
                if p.device != total.device:
                    p = p.to(total.device)
                total = total + p
        return total  # type: ignore[return-value]

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        for m in self.methods:
            m.before_task(model, train_loader, val_loader)

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        for m in self.methods:
            m.after_task(model, train_loader, val_loader)

    # ---- Hook opcional: encadena posibles wrappers de DataLoader (p.ej. Rehearsal) ----
    def prepare_train_loader(self, train_loader):
        """
        Si los submétodos exponen `prepare_train_loader`, los aplica en cascada:
        loader_out = m_k(...(m_2(m_1(train_loader)))).
        """
        loader = train_loader
        for m in self.methods:
            if hasattr(m, "prepare_train_loader"):
                loader = m.prepare_train_loader(loader)  # type: ignore[attr-defined]
        return loader
