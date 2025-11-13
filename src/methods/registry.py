# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List, Dict, Any
from torch import nn
import torch

from .base import BaseMethod
from .naive import Naive
from .ewc import EWCMethod
from .rehearsal import RehearsalMethod
from .composite import CompositeMethod

# SNN methods (cada uno hereda BaseMethod)
from .sa_snn import SASNN
from .as_snn import AS_SNN
from .sca_snn import SCA_SNN
from .colanet import CoLaNET

# ------------------ validación y filtrado ------------------

ALLOWED_KEYS: Dict[str, set[str]] = {
    "ewc": {
        "lambd", "fisher_batches",
        # NUEVOS parámetros de precisión/rendimiento/orientación
        "fisher_precision", "fisher_amp_dtype", "permute_policy",
    },
    "rehearsal": {
        "buffer_size", "replay_ratio",
        # NUEVOS
        "compress_mode", "pin_memory", "max_total_bs",
    },
    "sa-snn": {
        "attach_to", "k", "tau", "th_min", "th_max", "p", "vt_scale",
        "flatten_spatial", "assume_binary_spikes", "reset_counters_each_task",
        "ema_beta", "update_on_eval",
    },
    "as-snn": {
        "measure_at", "attach_to", "gamma_ratio", "lambda_a", "ema",
        "do_synaptic_scaling", "scale_clip", "scale_bias", "penalty_mode",
        "activity_verbose", "activity_every", "eps", "name_suffix",
    },
    "sca-snn": {
        "attach_to", "flatten_spatial", "num_bins", "anchor_batches",
        "bin_lo", "bin_hi", "max_per_bin", "beta", "bias",
        "soft_mask_temp", "habit_decay", "verbose", "log_every",
        # NUEVOS
        "target_active_frac", "T",
    },
    "naive": set(),
    "colanet": {"attach_to", "flatten_spatial"},
}

# Requisitos estrictos por método (el resto tiene defaults sensatos)
REQUIRED: Dict[str, set[str]] = {
    "ewc": {"lambd"},
}

GENERIC_TO_DROP = {
    "T", "epochs", "batch_size", "lr",
    "es_patience", "es_min_delta",
    "compile", "compile_mode", "amp",
    "encoder", "gain",
}


def _filter_kwargs_for(method_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    method = method_name.lower().strip()
    allowed = ALLOWED_KEYS.get(method, set())
    return {k: v for k, v in kwargs.items() if (k in allowed) and (k not in GENERIC_TO_DROP)}


def _validate_required(method_name: str, kwargs: Dict[str, Any]) -> None:
    req = REQUIRED.get(method_name.lower().strip(), set())
    missing = [k for k in req if k not in kwargs]
    if missing:
        ms = ", ".join(missing)
        raise AssertionError(f"Faltan parámetros requeridos para '{method_name}': {ms}")


# ------------------ fábrica ------------------

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
    Todos los métodos aceptan device/loss_fn en __init__ y (si lo necesitan)
    el model se inyecta en before_task.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    lname = name.lower().strip()

    # elimina claves genéricas no pertinentes al método
    for k in list(method_kwargs.keys()):
        if k in GENERIC_TO_DROP:
            method_kwargs.pop(k, None)

    def _build_base(main: str) -> BaseMethod:
        m = main.lower().strip()

        if m == "naive":
            return Naive()

        if m == "ewc":
            assert loss_fn is not None, "EWC requiere 'loss_fn'"
            ek = _filter_kwargs_for("ewc", method_kwargs)
            _validate_required("ewc", ek)
            return EWCMethod(device=device, loss_fn=loss_fn, **ek)

        if m == "rehearsal":
            rk = _filter_kwargs_for("rehearsal", method_kwargs)
            return RehearsalMethod(device=device, loss_fn=loss_fn, **rk)

        if m == "sa-snn":
            sk = _filter_kwargs_for("sa-snn", method_kwargs)
            return SASNN(device=device, loss_fn=loss_fn, **sk)

        if m == "as-snn":
            ak = _filter_kwargs_for("as-snn", method_kwargs)
            return AS_SNN(device=device, loss_fn=loss_fn, **ak)

        if m == "sca-snn":
            ck = _filter_kwargs_for("sca-snn", method_kwargs)
            return SCA_SNN(device=device, loss_fn=loss_fn, **ck)

        if m == "colanet":
            ck = _filter_kwargs_for("colanet", method_kwargs)
            return CoLaNET(device=device, loss_fn=loss_fn, **ck)

        raise ValueError(f"Método desconocido: {main}")

    if "+" in lname:
        parts: List[str] = [p.strip() for p in lname.split("+") if p.strip()]
        bases: List[BaseMethod] = [_build_base(p) for p in parts]
        return CompositeMethod(bases)

    return _build_base(lname)