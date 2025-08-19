# src/bench.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, torch
from contextlib import nullcontext
import src.training as training

def to_5d(xb, encoder: str, T: int, gain: float, device):
    """Devuelve (x5d, used_runtime) desde 4D o 5D."""
    if xb.ndim == 5:
        return xb.permute(1,0,2,3,4).contiguous(), False
    elif xb.ndim == 4:
        training.set_runtime_encode(mode=encoder, T=T, gain=gain, device=device)
        x5d = training._permute_if_needed(xb)  # encode+permute
        return x5d, True
    else:
        raise RuntimeError(f"Batch shape inesperada: {xb.shape}")

def forward_once_ms(model, x5d, device, use_amp=True):
    """Mide un forward único (ms)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.amp.autocast('cuda', enabled=(use_amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), ctx:
        _ = model(x5d.to(device, non_blocking=True))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0

def loop_gpu_only_its(model, x5d, device, iters=100, use_amp=True):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.amp.autocast('cuda', enabled=(use_amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), ctx:
        x5d_dev = x5d.to(device, non_blocking=True)
        for _ in range(iters):
            _ = model(x5d_dev)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return iters / (time.perf_counter() - t0)

def pipeline_its(model, loader, device, iters=100, use_amp=True, encoder=None, T=None, gain=None):
    """Itera loader+modelo (activa runtime encode si el loader es 4D)."""
    it = iter(loader)
    try:
        xb0, _ = next(it)
    except StopIteration:
        return float('nan')

    used_rt = False
    if xb0.ndim == 4:
        training.set_runtime_encode(mode=encoder, T=T, gain=gain, device=device)
        used_rt = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.amp.autocast('cuda', enabled=(use_amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), ctx:
        # primero
        x = xb0.permute(1,0,2,3,4).contiguous() if xb0.ndim==5 else training._permute_if_needed(xb0)
        _ = model(x.to(device, non_blocking=True))
        done = 1
        # resto
        while done < iters:
            try:
                xb, _ = next(it)
            except StopIteration:
                it = iter(loader)
                xb, _ = next(it)
            x = xb.permute(1,0,2,3,4).contiguous() if xb.ndim==5 else training._permute_if_needed(xb)
            _ = model(x.to(device, non_blocking=True))
            done += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    its = iters / (time.perf_counter() - t0)

    if used_rt:
        training.set_runtime_encode(None)
    return its
