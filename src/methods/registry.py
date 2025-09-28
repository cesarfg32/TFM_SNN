# src/methods/registry.py
from __future__ import annotations
from typing import Optional
from torch import nn
import torch

from .api import ContinualMethod
from .naive import Naive
from .ewc import EWCMethod
from .rehearsal import RehearsalMethod, RehearsalConfig
from .composite import CompositeMethod
# NUEVOS:
from .sa_snn import SA_SNN
from .as_snn import AS_SNN
from .sca_snn import SCA_SNN
from .colanet import CoLaNET

def build_method(
    name: str,
    model: nn.Module,
    *,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    **method_kwargs,
) -> ContinualMethod:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    lname = name.lower()

    def _build_base(main: str) -> ContinualMethod:
        if main == "naive":
            return Naive()
        if main == "ewc":
            assert loss_fn is not None, "EWC requiere 'loss_fn'"
            lam = method_kwargs.get("lam", None)
            assert lam is not None, "EWC requiere 'lam' en method_kwargs"
            fisher_batches = int(method_kwargs.get("fisher_batches", 100))
            return EWCMethod(model, float(lam), fisher_batches, loss_fn, device)
        if main == "rehearsal":
            cfg = RehearsalConfig(
                buffer_size=int(method_kwargs.get("buffer_size", 10_000)),
                replay_ratio=float(method_kwargs.get("replay_ratio", 0.2)),
            )
            return RehearsalMethod(cfg)
        # NUEVOS:
        if main == "sa-snn":
            return SA_SNN(**method_kwargs)
        if main == "as-snn":
            return AS_SNN(**method_kwargs)
        if main == "sca-snn":
            return SCA_SNN(**method_kwargs)
        if main == "colanet":
            return CoLaNET(**method_kwargs)
        raise ValueError(f"Método desconocido: {main}")

    # Soporta "<main>+ewc" y también "ewc+<main>"
    if "+" in lname:
        parts = [p.strip() for p in lname.split("+") if p.strip()]
        bases = []
        ewc_cfg = {}
        for p in parts:
            if p == "ewc":
                lam = method_kwargs.get("lam", method_kwargs.get("ewc_lam", None))
                assert lam is not None, "Composite +EWC requiere 'lam' (o 'ewc_lam')"
                assert loss_fn is not None, "Composite +EWC requiere 'loss_fn'"
                fisher_batches = int(method_kwargs.get("fisher_batches", 100))
                bases.append(EWCMethod(model, float(lam), fisher_batches, loss_fn, device))
            else:
                bases.append(_build_base(p))
        return CompositeMethod(bases)

    # método puro
    return _build_base(lname)
