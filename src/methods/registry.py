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
    fisher_batches: int = 100,
    **method_kwargs,
) -> ContinualMethod:
    """
    Builder unificado:
      - "naive"
      - "ewc"                  -> requiere lam en method_kwargs["lam"]
      - "rehearsal"            -> buffer_size, replay_ratio en kwargs
      - "rehearsal+ewc"        -> añade ewc encima (ewc_lam en kwargs)
      - futuros: "<main>+ewc"
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
            return EWCMethod(model, float(lam), int(fisher_batches), loss_fn, device)
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
        ewc_lam = method_kwargs.get("ewc_lam", None)
        assert ewc_lam is not None, "Composite +EWC requiere 'ewc_lam' en method_kwargs"
        ewc = EWCMethod(model, float(ewc_lam), int(fisher_batches), loss_fn, device)
        return CompositeMethod([base, ewc])

    # método puro
    return _build_base(lname)
