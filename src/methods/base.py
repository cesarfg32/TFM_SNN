# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC
from typing import Optional, Any, Dict
import torch
from torch import nn

class BaseMethod(ABC):
    """
    Contrato mínimo para métodos de Aprendizaje Continuo.
    Hooks no-op por defecto. `penalty()` devuelve un tensor escalar en el device de la instancia.
    """
    name: str = "method"
    inner_verbose: bool = False
    inner_every: int = 50

    def __init__(self, *, device: Optional[torch.device] = None, loss_fn: Optional[nn.Module] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.loss_fn = loss_fn

    # ---- Hooks opcionales ----
    def prepare_train_loader(self, train_loader): return train_loader
    def before_task(self, model: nn.Module, train_loader, val_loader) -> None: ...
    def after_task(self,  model: nn.Module, train_loader, val_loader) -> None: ...
    def before_epoch(self, model: nn.Module, epoch: int) -> None: ...
    def after_epoch(self,  model: nn.Module, epoch: int) -> None: ...
    def before_batch(self, model: nn.Module, batch) -> None: ...
    def after_batch(self,  model: nn.Module, batch, loss) -> None: ...

    # ---- Regularización ----
    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    # ---- Estado opcional ----
    def get_state(self) -> Dict[str, Any]: return {}
    def load_state(self, state: Dict[str, Any]) -> None: ...
    def detach(self) -> None: ...
    def tunable(self) -> Dict[str, Any]: return {}
