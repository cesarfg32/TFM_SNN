#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Tuple

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - entorno sin PyTorch
    print(
        "[SKIP] PyTorch no está instalado: esta prueba de humo requiere torch. "
        "Instala torch/torchvision según tu CUDA o CPU para ejecutarla."
    )
    raise SystemExit(2) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.methods.registry import build_method
from src.nn_io import _align_target_shape

def _dev(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def synthetic_loader(batches=3, B=8, T=8, H=66, W=200) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for i in range(batches):
        # alterna (B,...) y (T,B,...)
        if (i % 2) == 0:
            x = torch.rand(B,1,H,W)
        else:
            x = torch.rand(T,B,1,H,W)
        y = torch.rand(B)*2.0 - 1.0
        yield x, y

class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(8,16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32,1)
    def forward(self, x):
        if x.ndim==5: x=x.mean(0)  # colapsa T en la cabeza dummy
        return self.fc(self.conv(x).flatten(1))

def _maybe_build_method(name: str, base_model: nn.Module, params: Dict[str, Any]):
    dev=_dev()
    model=base_model.to(dev).eval()
    loss=nn.MSELoss()
    return build_method(name=name, model=model, loss_fn=loss, device=dev, **params)

def _run_one(name: str, params: Dict[str, Any]) -> bool:
    dev=_dev()
    model=TinyHead().to(dev).eval()
    method = _maybe_build_method(name, model, params)
    mse=mae=0.0; n=0
    for xb,yb in synthetic_loader():
        xb=xb.to(dev); yb=yb.to(dev)
        y_hat_raw = method.impl(model, xb) if hasattr(method, "impl") and callable(getattr(method, "impl", None)) else model(xb)  # legacy safety
        y_hat = y_hat_raw if isinstance(y_hat_raw, torch.Tensor) else (y_hat_raw[0] if isinstance(y_hat_raw,(list,tuple)) else y_hat_raw)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = model(xb)
        yb2 = _align_target_shape(y_hat, yb).to(device=y_hat.device, dtype=y_hat.dtype)
        mse += torch.mean((y_hat - yb2)**2).item()
        mae += torch.mean(torch.abs(y_hat - yb2)).item()
        n+=1
    print(f"[{name}] OK | mse≈{mse/max(1,n):.4g} mae≈{mae/max(1,n):.4g}")
    return True

if __name__ == "__main__":
    ok=True
    cases = [
        ("ewc", {"lam":3e6, "fisher_batches":5}),
        ("sca-snn", {"attach_to":"fc","num_bins":10,"beta":0.55,"soft_mask_temp":0.3,"anchor_batches":2}),
        ("as-snn", {"attach_to":"conv.2","gamma_ratio":0.4,"lambda_a":0.5,"measure_at":"modules"}),
        ("sa-snn", {"attach_to":"fc","k":8,"tau":28,"vt_scale":1.33,"p":2_000_000,
                    "th_min":1.0,"th_max":2.0,"flatten_spatial":False}),
        ("rehearsal", {"buffer_size":64,"replay_ratio":0.25}),
        ("ewc+sca-snn", {"lam":1e6,"fisher_batches":3,"attach_to":"fc","num_bins":10,"beta":0.4}),
    ]
    for name,params in cases:
        ok &= _run_one(name, params)
    sys.exit(0 if ok else 1)
