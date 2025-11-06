# src/methods/registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List
from torch import nn
import torch

from .api import ContinualMethod
from .naive import Naive
from .ewc import EWCMethod
from .rehearsal import RehearsalMethod, RehearsalConfig
from .composite import CompositeMethod

# Nuevos
from .sa_snn import SASNN, SASNNConfig
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
    """
    Construye el método de aprendizaje continuo a partir del nombre.
    Soporta combinaciones tipo "<main>+ewc" o "ewc+<main>".
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    lname = name.lower().strip()

    # --- Filtrado de kwargs genéricos que pasan desde el preset pero el método no espera ---
    # Ej. muchos presets incluyen 'T' del encoder. Evitamos TypeError en constructores.
    generic_to_drop = {"T"}  # añade aquí si hay otros genéricos (ej. 'epochs', etc.) que "cuelen"
    for k in list(method_kwargs.keys()):
        if k in generic_to_drop:
            method_kwargs.pop(k, None)

    def _attach_name_and_hooks(obj: ContinualMethod, nm: str) -> ContinualMethod:
        # Garantiza .name
        if not hasattr(obj, "name"):
            try:
                setattr(obj, "name", nm)
            except Exception:
                pass
        # Hooks no-op por si faltan (para evitar AttributeError)
        for hook in ("before_task", "after_task", "before_epoch", "after_epoch",
                     "before_batch", "after_batch", "close", "state_dict", "load_state_dict", "penalty"):
            if not hasattr(obj, hook):
                if hook == "penalty":
                    setattr(obj, hook, (lambda self=obj: 0.0))
                elif hook in ("state_dict",):
                    setattr(obj, hook, (lambda self=obj: {}))
                elif hook in ("load_state_dict",):
                    setattr(obj, hook, (lambda state, self=obj: None))
                else:
                    setattr(obj, hook, (lambda *a, **kw: None))
        return obj

    def _build_base(main: str) -> ContinualMethod:
        if main == "naive":
            return _attach_name_and_hooks(Naive(), "naive")

        if main == "ewc":
            assert loss_fn is not None, "EWC requiere 'loss_fn'"
            lam = method_kwargs.get("lam", method_kwargs.get("ewc_lam", None))
            assert lam is not None, "EWC requiere 'lam' en method_kwargs (o 'ewc_lam')"
            fisher_batches = int(method_kwargs.get("fisher_batches", 100))
            return _attach_name_and_hooks(EWCMethod(model, float(lam), fisher_batches, loss_fn, device), "ewc")

        if main == "rehearsal":
            cfg = RehearsalConfig(
                buffer_size=int(method_kwargs.get("buffer_size", 10_000)),
                replay_ratio=float(method_kwargs.get("replay_ratio", 0.2)),
            )
            return _attach_name_and_hooks(RehearsalMethod(cfg), "rehearsal")

        # --- Nuevos métodos SNN ---
        if main == "sa-snn":
            cfg = SASNNConfig(**method_kwargs)
            obj = SASNN(model=model, cfg=cfg, device=device)
            return _attach_name_and_hooks(obj, "sa-snn")

        if main == "as-snn":
            return _attach_name_and_hooks(AS_SNN(**method_kwargs), "as-snn")

        if main == "sca-snn":
            return _attach_name_and_hooks(SCA_SNN(**method_kwargs), "sca-snn")

        if main == "colanet":
            return _attach_name_and_hooks(CoLaNET(**method_kwargs), "colanet")

        raise ValueError(f"Método desconocido: {main}")

    # Soporta "<main>+ewc" y también "ewc+<main>"
    if "+" in lname:
        parts: List[str] = [p.strip() for p in lname.split("+") if p.strip()]
        bases: List[ContinualMethod] = []

        for p in parts:
            if p == "ewc":
                lam = method_kwargs.get("lam", method_kwargs.get("ewc_lam", None))
                assert lam is not None, "Composite +EWC requiere 'lam' (o 'ewc_lam')"
                assert loss_fn is not None, "Composite +EWC requiere 'loss_fn'"
                fisher_batches = int(method_kwargs.get("fisher_batches", 100))
                bases.append(_attach_name_and_hooks(EWCMethod(model, float(lam), fisher_batches, loss_fn, device), "ewc"))
            else:
                bases.append(_build_base(p))

        comp = CompositeMethod(bases)
        if not hasattr(comp, "name"):
            comp_name = "+".join(getattr(b, "name", b.__class__.__name__.lower()) for b in bases)
            _attach_name_and_hooks(comp, comp_name)
        return comp

    # Método simple
    return _build_base(lname)
