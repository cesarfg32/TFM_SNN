# src/methods/registry.py
from __future__ import annotations
from typing import Optional
from torch import nn
import torch

from .api import ContinualMethod
from .naive import Naive
from .ewc import EWCMethod              # <- usa el wrapper interno al módulo ewc
from .composite import CompositeMethod

def build_single_method(
    name: str,
    model: nn.Module,
    *,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    lambd: Optional[float] = None,
    fisher_batches: int = 100,
) -> ContinualMethod:
    name = name.lower()
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    if name == "naive":
        return Naive()

    if name == "ewc":
        assert lambd is not None, "EWC requiere 'lambd'"
        assert loss_fn is not None, "EWC requiere 'loss_fn'"
        return EWCMethod(model, lambd, fisher_batches, loss_fn, device)

    # ⬇️ En el futuro: enchufar AS-SNN, SA-SNN, etc. (cuando queramos)
    # if name == "as_snn": return ASSNNAdapter(...)
    # if name == "sa_snn": return SASNNAdapter(...)
    raise ValueError(f"Método desconocido (por ahora): {name}")

def build_method_with_optional_ewc(
    main_method: str,
    model: nn.Module,
    *,
    loss_fn: Optional[nn.Module],
    device: Optional[torch.device],
    fisher_batches: int,
    ewc_lam: Optional[float],   # si no es None, se añade EWC en composición
) -> ContinualMethod:
    base = build_single_method(main_method, model, loss_fn=loss_fn, device=device)
    if ewc_lam is None:
        return base
    ewc = EWCMethod(model, ewc_lam, fisher_batches, loss_fn, device or torch.device("cpu"))
    return CompositeMethod([base, ewc])
