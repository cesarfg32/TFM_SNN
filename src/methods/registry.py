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

def build_method(
    name: str,
    model: nn.Module,
    *,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    **method_kwargs,
) -> ContinualMethod:
    """
    Builder unificado:
      - "naive"
      - "ewc"                 -> requiere lam en method_kwargs["lam"]; fisher_batches opcional
      - "rehearsal"           -> buffer_size, replay_ratio en kwargs
      - "<main>+ewc"          -> añade EWC encima (lam en kwargs; alias ewc_lam aceptado)
    """
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

        raise ValueError(f"Método desconocido: {main}")

    if "+ewc" in lname:
        main = lname.replace("+ewc", "")
        base = _build_base(main)

        assert loss_fn is not None, "Composite +EWC requiere 'loss_fn'"
        # Acepta 'lam' (preferido) y 'ewc_lam' como alias de compatibilidad.
        lam = method_kwargs.get("lam", method_kwargs.get("ewc_lam", None))
        assert lam is not None, "Composite +EWC requiere 'lam' (o 'ewc_lam') en method_kwargs"
        fisher_batches = int(method_kwargs.get("fisher_batches", 100))

        ewc = EWCMethod(model, float(lam), fisher_batches, loss_fn, device)
        return CompositeMethod([base, ewc])

    # método puro
    return _build_base(lname)
