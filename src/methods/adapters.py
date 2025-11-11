# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from torch import nn
from .base import BaseMethod

def _scalarize(x: Any) -> Any:
    """Convierte penalty en escalar (float o tensor escalar)."""
    try:
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, torch.Tensor): return x if x.ndim == 0 else x.mean()
        if isinstance(x, (list, tuple)):
            vals = [_scalarize(v) for v in x]
            vals = [v for v in vals if v is not None]
            if not vals: return 0.0
            acc = 0.0
            for v in vals:
                acc = acc + (v if isinstance(v, torch.Tensor) else float(v))
            return acc
        if isinstance(x, dict):
            vals = [_scalarize(v) for v in x.values()]
            return _scalarize(vals)
        return float(x)
    except Exception:
        return 0.0

class MethodAdapter(BaseMethod):
    """
    Envuelve cualquier implementación 'impl' y la adapta al contrato BaseMethod:
    - Delegación de hooks si existen
    - penalty() garantizado escalar
    - name/inner_verbose/inner_every normalizados
    """
    def __init__(self, impl: Any, *, name: Optional[str] = None,
                 device: Optional[torch.device] = None, loss_fn: Optional[nn.Module] = None):
        super().__init__(device=device, loss_fn=loss_fn)
        self.impl = impl
        self.name = getattr(impl, "name", None) or name or impl.__class__.__name__.lower()
        self.inner_verbose = getattr(impl, "inner_verbose", False)
        self.inner_every   = int(getattr(impl, "inner_every", 50))

    def prepare_train_loader(self, train_loader):
        fn = getattr(self.impl, "prepare_train_loader", None)
        if callable(fn):
            try:
                return fn(train_loader)
            except Exception:
                return train_loader
        return train_loader

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        fn = getattr(self.impl, "before_task", None)
        if callable(fn):
            try: fn(model, train_loader, val_loader)
            except Exception: pass

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        fn = getattr(self.impl, "after_task", None)
        if callable(fn):
            try: fn(model, train_loader, val_loader)
            except Exception: pass

    def before_epoch(self, model: nn.Module, epoch: int) -> None:
        fn = getattr(self.impl, "before_epoch", None)
        if callable(fn):
            try: fn(model, epoch)
            except Exception: pass

    def after_epoch(self, model: nn.Module, epoch: int) -> None:
        fn = getattr(self.impl, "after_epoch", None)
        if callable(fn):
            try: fn(model, epoch)
            except Exception: pass

    def before_batch(self, model: nn.Module, batch) -> None:
        fn = getattr(self.impl, "before_batch", None)
        if callable(fn):
            try: fn(model, batch)
            except Exception: pass

    def after_batch(self, model: nn.Module, batch, loss) -> None:
        fn = getattr(self.impl, "after_batch", None)
        if callable(fn):
            try: fn(model, batch, loss)
            except Exception: pass

    def penalty(self):
        p = getattr(self.impl, "penalty", None)
        out = _scalarize(p()) if callable(p) else 0.0
        if isinstance(out, torch.Tensor):
            out = out.to(self.device, non_blocking=True)
            return out.mean() if out.ndim > 0 else out
        return float(out)
