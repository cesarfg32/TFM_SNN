# src/methods/helpers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

def ensure_attach_to(model: nn.Module, kwargs: Dict, default: str = "f6") -> Dict:
    """
    Garantiza kwargs['attach_to'] apuntando a un submódulo existente.
    Estrategia:
      1) Si attach_to existe y está en named_modules -> OK.
      2) Si no, usa 'default' si existe (por convención de tu PilotNet).
      3) Si no, elige el último nn.Linear del modelo.
      4) Si no, elige el último submódulo registrado.
    """
    mods = dict(model.named_modules())
    name = str(kwargs.get("attach_to", "")).strip()
    if name and name in mods:
        return kwargs

    if default in mods:
        kwargs["attach_to"] = default
        return kwargs

    last_linear = None
    for n, m in mods.items():
        if isinstance(m, nn.Linear):
            last_linear = n
    if last_linear:
        kwargs["attach_to"] = last_linear
        return kwargs

    # fallback extremo: último submódulo (evita crash, aunque sea menos semántico)
    if mods:
        kwargs["attach_to"] = list(mods.keys())[-1]
    return kwargs


def sanitize_as_snn_kwargs(kwargs: Dict) -> Dict:
    """Corrige penalty_mode inválido y aplica defaults razonables."""
    pm = str(kwargs.get("penalty_mode", "l2")).lower().strip()
    if pm not in ("l1", "l2"):
        kwargs["penalty_mode"] = "l2"
    return kwargs
