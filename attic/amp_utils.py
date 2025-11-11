# src/amp_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
from torch import nn

# Capas "seguras" para castear de dtype (no BN, no capas raras)
_ALLOWLIST = (nn.Conv2d, nn.Linear)

def cast_linear_conv(module: nn.Module, dtype: torch.dtype) -> None:
    """Castea en sitio pesos y bias de Conv/Linear al dtype dado."""
    for m in module.modules():
        if isinstance(m, _ALLOWLIST):
            m.to(dtype=dtype)
