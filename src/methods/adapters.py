# src/methods/adapters.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from torch import nn
from .base import BaseMethod

def _scalarize(x: Any) -> Any:
    """
    Intenta convertir salida de penalty a escalar (float o tensor escalar).
    Admite: float/int, tensor escalar, tensor -> mean(), lista/tupla/dict -> suma segura.
    """
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                return x
            # por si alguien devolviera un vector de penalizaciones
            return x.mean()
        if isinstance(x, (list, tuple)):
            vals = [_scalarize(v) for v in x]
            vals = [v for v in vals if v is not None]
            if not vals:
                return 0.0
            # suma cuidando tensores/float
            acc = 0.0
            for v in vals:
                if isinstance(v, torch.Tensor):
                    acc = acc + v
                else:
                    acc = acc + float(v)
            return acc
        if isinstance(x, dict):
            vals = [_scalarize(v) for v in x.values()]
            return _scalarize(vals)
        # fallback
        return float(x)
    except Exception:
        return 0.0

class MethodAdapter(BaseMethod):
    """
    Envuelve cualquier implementación 'impl' y la adapta al contrato BaseMethod:
    - Hooks delegados si existen
    - penalty() garantizado escalar (tensor escalar o float)
    - name/inner_verbose/inner_every normalizados
    - estado opcional (get_state/load_state/state_dict)
    """
    def __init__(self, impl: Any, *, name: Optional[str] = None,
                 device: Optional[torch.device] = None, loss_fn: Optional[nn.Module] = None):
        super().__init__(device=device, loss_fn=loss_fn)
        self.impl = impl
        # nombre: preferimos .name; si no, el proporcionado; si no, clase
        self.name = (
            getattr(impl, "name", None)
            or name
            or impl.__class__.__name__.lower()
        )
        # logging interno si el impl ya lo define
        self.inner_verbose = getattr(impl, "inner_verbose", False)
        self.inner_every   = int(getattr(impl, "inner_every", 50))

    # ---- Delegaciones seguras de hooks ----
    def prepare_train_loader(self, train_loader):
        if hasattr(self.impl, "prepare_train_loader"):
            try:
                return self.impl.prepare_train_loader(train_loader)
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
        if callable(p):
            out = p()
        else:
            out = 0.0
        out = _scalarize(out)
        # devolvemos float o tensor escalar; training ya castea a dtype/device de y_hat
        return out

    def get_state(self) -> Dict[str, Any]:
        for key in ("get_state", "get_activity_state", "state_dict"):
            fn = getattr(self.impl, key, None)
            if callable(fn):
                try:
                    st = fn()
                    # state_dict de nn.Module devuelve OrderedDict; está bien
                    return st if isinstance(st, dict) else {}
                except Exception:
                    pass
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        for key in ("load_state", "load_state_dict"):
            fn = getattr(self.impl, key, None)
            if callable(fn):
                try:
                    fn(state)
                    return
                except Exception:
                    pass

    def detach(self) -> None:
        for key in ("detach", "close"):
            fn = getattr(self.impl, key, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass

    def tunable(self) -> Dict[str, Any]:
        fn = getattr(self.impl, "tunable", None)
        if callable(fn):
            try:
                d = fn() or {}
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        return {}
