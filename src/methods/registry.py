# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, List, Dict, Any
from torch import nn
import torch

from .base import BaseMethod
from .naive import Naive
from .ewc import EWCMethod
from .rehearsal import RehearsalMethod, RehearsalConfig
from .composite import CompositeMethod

# SNN methods (cada uno hereda BaseMethod)
from .sa_snn import SASNN, SASNNConfig
from .as_snn import AS_SNN
from .sca_snn import SCA_SNN
from .colanet import CoLaNET

EWC_KEYS = {"lam", "ewc_lam", "fisher_batches"}

ALLOWED_KEYS: Dict[str, set[str]] = {
    "sa-snn": {
        "attach_to", "k", "tau", "th_min", "th_max", "p", "vt_scale",
        "flatten_spatial", "assume_binary_spikes", "reset_counters_each_task",
+        "ema_beta","update_on_eval"
    },
    "as-snn": {
        "measure_at", "attach_to", "gamma_ratio", "lambda_a", "ema",
        "do_synaptic_scaling", "scale_clip", "scale_bias", "penalty_mode",
        "activity_verbose", "activity_every", "eps", "name_suffix",
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

def build_method(
    name: str,
    model: nn.Module,
    *,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    **method_kwargs,
) -> BaseMethod:
    """
    Construye el método a partir del nombre. Soporta composites "a+b".
    Sin adapters: cada método hereda BaseMethod.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    lname = name.lower().strip()

    for k in list(method_kwargs.keys()):
        if k in GENERIC_TO_DROP:
            method_kwargs.pop(k, None)

    def _build_base(main: str) -> BaseMethod:
        m = main.lower().strip()

        if m == "naive":
            return Naive()

        if m == "ewc":
            assert loss_fn is not None, "EWC requiere 'loss_fn'"
            ek = _ewc_kwargs(method_kwargs)
            lam = ek.get("lam", None)
            assert lam is not None, "EWC requiere 'lam' (o 'ewc_lam')"
            fisher_batches = int(ek.get("fisher_batches", 100))
            return EWCMethod(model, float(lam), fisher_batches, loss_fn, device)

        if m == "rehearsal":
            rk = _filter_kwargs_for("rehearsal", method_kwargs)
            cfg = RehearsalConfig(
                buffer_size=int(rk.get("buffer_size", 10_000)),
                replay_ratio=float(rk.get("replay_ratio", 0.2)),
            )
            obj = RehearsalMethod(cfg)
            obj.device = device
            obj.loss_fn = loss_fn
            return obj

        if m == "sa-snn":
            sk = _filter_kwargs_for("sa-snn", method_kwargs)
            cfg = SASNNConfig(**sk)
            return SASNN(model=model, cfg=cfg, device=device)

        if m == "as-snn":
            ak = _filter_kwargs_for("as-snn", method_kwargs)
            obj = AS_SNN(**ak)
            obj.device = device
            obj.loss_fn = loss_fn
            return obj

        if m == "sca-snn":
            ck = _filter_kwargs_for("sca-snn", method_kwargs)
            obj = SCA_SNN(**ck)
            obj.device = device
            obj.loss_fn = loss_fn
            return obj

        if m == "colanet":
            ck = _filter_kwargs_for("colanet", method_kwargs)
            obj = CoLaNET(**ck)
            obj.device = device
            obj.loss_fn = loss_fn
            return obj

        raise ValueError(f"Método desconocido: {main}")

    if "+" in lname:
        parts: List[str] = [p.strip() for p in lname.split("+") if p.strip()]
        bases: List[BaseMethod] = [ _build_base(p) for p in parts ]
        return CompositeMethod(bases)

    return _build_base(lname)
