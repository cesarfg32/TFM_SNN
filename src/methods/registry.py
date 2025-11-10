# src/methods/registry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, List, Dict, Any
from torch import nn
import torch

from .base import BaseMethod
from .adapters import MethodAdapter
from .naive import Naive
from .ewc import EWCMethod
from .rehearsal import RehearsalMethod, RehearsalConfig
from .composite import CompositeMethod

# SNN methods (implementación existente)
from .sa_snn import SASNN, SASNNConfig
from .as_snn import AS_SNN
from .sca_snn import SCA_SNN
from .colanet import CoLaNET

# --------------------------------------------------------------------------------------
# Filtros de kwargs por método (evita pasar lam/fisher a SA/AS/SCA, etc.)
# --------------------------------------------------------------------------------------
EWC_KEYS = {"lam", "ewc_lam", "fisher_batches"}

ALLOWED_KEYS: Dict[str, set[str]] = {
    "sa-snn": {
        "attach_to", "k", "tau", "th_min", "th_max", "p", "vt_scale",
        "flatten_spatial", "assume_binary_spikes", "reset_counters_each_task"
    },
    "as-snn": {
        "measure_at", "attach_to", "gamma_ratio", "lambda_a", "ema",
        "do_synaptic_scaling", "scale_clip", "scale_bias", "penalty_mode"
    },
    "sca-snn": {
        "attach_to", "flatten_spatial", "num_bins", "anchor_batches",
        "bin_lo", "bin_hi", "max_per_bin", "beta", "bias",
        "soft_mask_temp", "habit_decay", "verbose", "log_every"
    },
    "rehearsal": {"buffer_size", "replay_ratio"},
    "naive": set(),
    "colanet": {"attach_to", "flatten_spatial"},
}

GENERIC_TO_DROP = {
    "T", "epochs", "batch_size", "lr", "es_patience", "es_min_delta",
    "compile", "compile_mode", "amp", "encoder", "gain"
}

def _filter_kwargs_for(method_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    method = method_name.lower().strip()
    allowed = ALLOWED_KEYS.get(method, set())
    return {k: v for k, v in kwargs.items() if (k in allowed) and (k not in GENERIC_TO_DROP)}

def _ewc_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in kwargs.items() if k in EWC_KEYS}
    if "lam" not in out and "ewc_lam" in out:
        out["lam"] = out.pop("ewc_lam")
    return out

def _wrap(obj: BaseMethod | Any, *, loss_fn=None, device=None, name_hint: Optional[str] = None) -> BaseMethod:
    """
    Devuelve un BaseMethod:
    - Si ya es BaseMethod, lo retorna tal cual.
    - Si no, lo envuelve con MethodAdapter (delegación).
    """
    if isinstance(obj, BaseMethod):
        return obj
    return MethodAdapter(obj, name=name_hint, device=device, loss_fn=loss_fn)

# --------------------------------------------------------------------------------------
# build_method: construcción de métodos y composites, con ADAPTACIÓN universal
# --------------------------------------------------------------------------------------
def build_method(
    name: str,
    model: nn.Module,
    *,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    **method_kwargs,
) -> BaseMethod:
    """
    Construye el método de aprendizaje continuo a partir del nombre.
    Soporta combinaciones tipo "<main>+ewc" o "ewc+<main>".
    Aplica:
      - filtrado de kwargs por método
      - adaptación universal a BaseMethod (MethodAdapter) para métodos legacy
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    lname = name.lower().strip()

    # Limpia genéricos
    for k in list(method_kwargs.keys()):
        if k in GENERIC_TO_DROP:
            method_kwargs.pop(k, None)

    def _build_base(main: str) -> BaseMethod:
        m = main.lower().strip()

        if m == "naive":
            return _wrap(Naive(), loss_fn=loss_fn, device=device, name_hint="naive")

        if m == "ewc":
            assert loss_fn is not None, "EWC requiere 'loss_fn'"
            ek = _ewc_kwargs(method_kwargs)
            lam = ek.get("lam", None)
            assert lam is not None, "EWC requiere 'lam' (o 'ewc_lam')"
            fisher_batches = int(ek.get("fisher_batches", 100))
            # EWCMethod ya hereda de BaseMethod
            return EWCMethod(model, float(lam), fisher_batches, loss_fn, device)

        if m == "rehearsal":
            rk = _filter_kwargs_for("rehearsal", method_kwargs)
            cfg = RehearsalConfig(
                buffer_size=int(rk.get("buffer_size", 10_000)),
                replay_ratio=float(rk.get("replay_ratio", 0.2)),
            )
            return _wrap(RehearsalMethod(cfg), loss_fn=loss_fn, device=device, name_hint="rehearsal")

        # --- SNN methods: se construyen como hasta ahora y se envuelven ---
        if m == "sa-snn":
            sk = _filter_kwargs_for("sa-snn", method_kwargs)
            cfg = SASNNConfig(**sk)
            obj = SASNN(model=model, cfg=cfg, device=device)
            return _wrap(obj, loss_fn=loss_fn, device=device, name_hint="sa-snn")

        if m == "as-snn":
            ak = _filter_kwargs_for("as-snn", method_kwargs)
            obj = AS_SNN(**ak)
            return _wrap(obj, loss_fn=loss_fn, device=device, name_hint="as-snn")

        if m == "sca-snn":
            ck = _filter_kwargs_for("sca-snn", method_kwargs)
            obj = SCA_SNN(**ck)
            return _wrap(obj, loss_fn=loss_fn, device=device, name_hint="sca-snn")

        if m == "colanet":
            ck = _filter_kwargs_for("colanet", method_kwargs)
            obj = CoLaNET(**ck)
            return _wrap(obj, loss_fn=loss_fn, device=device, name_hint="colanet")

        raise ValueError(f"Método desconocido: {main}")

    # --- Composite "<main>+ewc" y/o "ewc+<main>" ---
    if "+" in lname:
        parts: List[str] = [p.strip() for p in lname.split("+") if p.strip()]
        bases: List[BaseMethod] = []
        for p in parts:
            pl = p.lower()
            if pl == "ewc":
                ek = _ewc_kwargs(method_kwargs)
                lam = ek.get("lam", None)
                assert lam is not None, "Composite +EWC requiere 'lam' (o 'ewc_lam')"
                assert loss_fn is not None, "Composite +EWC requiere 'loss_fn'"
                fisher_batches = int(ek.get("fisher_batches", 100))
                bases.append(EWCMethod(model, float(lam), fisher_batches, loss_fn, device))
            elif pl == "sa-snn":
                sub = _filter_kwargs_for("sa-snn", method_kwargs)
                cfg = SASNNConfig(**sub)
                bases.append(_wrap(SASNN(model=model, cfg=cfg, device=device), loss_fn=loss_fn, device=device, name_hint="sa-snn"))
            elif pl == "as-snn":
                sub = _filter_kwargs_for("as-snn", method_kwargs)
                bases.append(_wrap(AS_SNN(**sub), loss_fn=loss_fn, device=device, name_hint="as-snn"))
            elif pl == "sca-snn":
                sub = _filter_kwargs_for("sca-snn", method_kwargs)
                bases.append(_wrap(SCA_SNN(**sub), loss_fn=loss_fn, device=device, name_hint="sca-snn"))
            elif pl == "rehearsal":
                sub = _filter_kwargs_for("rehearsal", method_kwargs)
                cfg = RehearsalConfig(
                    buffer_size=int(sub.get("buffer_size", 10_000)),
                    replay_ratio=float(sub.get("replay_ratio", 0.2)),
                )
                bases.append(_wrap(RehearsalMethod(cfg), loss_fn=loss_fn, device=device, name_hint="rehearsal"))
            elif pl == "naive":
                bases.append(_wrap(Naive(), loss_fn=loss_fn, device=device, name_hint="naive"))
            elif pl == "colanet":
                sub = _filter_kwargs_for("colanet", method_kwargs)
                bases.append(_wrap(CoLaNET(**sub), loss_fn=loss_fn, device=device, name_hint="colanet"))
            else:
                raise ValueError(f"Método desconocido dentro de composite: {p}")

        comp = CompositeMethod(bases)
        # Asegura .name
        if not hasattr(comp, "name"):
            comp.name = "+".join(getattr(b, "name", b.__class__.__name__.lower()) for b in bases)
        # Garantiza compat con BaseMethod
        return _wrap(comp, loss_fn=loss_fn, device=device, name_hint=comp.name)

    # Método simple
    return _build_base(lname)
