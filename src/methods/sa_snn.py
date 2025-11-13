# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseMethod

DeviceLike = Union[torch.device, str]


@dataclass
class SASNNConfig:
    attach_to: str = "auto"
    k: float = 8.0
    tau: float = 32.0
    vt_scale: float = 1.33
    th_min: float = 1.0
    th_max: float = 2.0
    p: int = 2_000_000
    ema_beta: Optional[float] = None
    assume_binary_spikes: bool = False
    flatten_spatial: bool = False
    reset_counters_each_task: bool = False
    update_on_eval: bool = False


class SASNN(BaseMethod):
    """SA-SNN (Sparse Activation por tracking de actividad)."""
    name = "sa-snn"

    def __init__(self, *, device: Optional[DeviceLike] = None, loss_fn=None, **kw):
        super().__init__(device=(torch.device(device) if device is not None else None), loss_fn=loss_fn)
        self.cfg = SASNNConfig(**{k: v for k, v in kw.items() if k in SASNNConfig.__annotations__})

        # Estado de enganche
        self._target: Optional[nn.Module] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # Buffers de actividad
        self._neurons_N: Optional[int] = None
        self._tr: Optional[Tensor] = None
        self._ema: Optional[Tensor] = None
        self._counters: Optional[Tensor] = None
        self._t_seen: int = 0
        self._vt_bias: Optional[Tensor] = None

        # τ y betas
        self._tau = float(max(1.0, self.cfg.tau))
        self._th_min = float(self.cfg.th_min)
        self._th_max = float(max(self.cfg.th_min, self.cfg.th_max))

        if self.cfg.ema_beta is not None:
            self._ema_beta = float(self.cfg.ema_beta)
        else:
            # aproximación a 1 - 1/p para EMA
            self._ema_beta = float(torch.exp(torch.tensor(-1.0 / max(1, self.cfg.p))).item())

    # ---- localización del target ----
    @staticmethod
    def _find_target_module(model: nn.Module, name: str) -> Optional[nn.Module]:
        if not name:
            return None
        if name == "auto":
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    return m
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    return m
            return None
        lookup: Dict[str, nn.Module] = dict(model.named_modules())
        return lookup.get(name, None)

    # ---- hooks ----
    def _forward_hook(self, module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> Tensor:
        out = output
        if not isinstance(out, torch.Tensor):
            return out
        if out.device != self.device:
            out = out.to(self.device, non_blocking=True)

        if out.dim() < 2:
            B = out.shape[0]
            out = out.view(B, 1)
        B, C = out.shape[:2]
        spatial = out.shape[2:]
        if self.cfg.flatten_spatial:
            HW = int(torch.prod(torch.tensor(spatial)).item()) if spatial else 1
            N = max(1, C * HW)
        else:
            N = max(1, C)

        if self._neurons_N is None:
            self._init_buffers(N)

        with torch.no_grad():
            S = self._reduce_activity(out.detach(), flatten_spatial=self.cfg.flatten_spatial)
            if self.cfg.assume_binary_spikes:
                S = (S > 0).to(out.dtype)

            is_train_mode = module.training or self.cfg.update_on_eval
            if is_train_mode:
                tr_next = self._tr - (self._tr / self._tau) + S
                tr_next = torch.nan_to_num(tr_next, nan=0.0, posinf=0.0, neginf=0.0)
                self._tr = tr_next
                self._ema.mul_(self._ema_beta).add_(S, alpha=(1.0 - self._ema_beta))
                self._ema[:] = torch.nan_to_num(self._ema, nan=0.0, posinf=0.0, neginf=0.0)
                self._counters.add_(S)
                self._t_seen += 1

            vt_bias = self._compute_vt_bias()
            score = torch.nan_to_num(self._tr - vt_bias, nan=0.0, posinf=0.0, neginf=0.0)
            kk = self._resolve_k(self.cfg.k, N)
            if kk >= N:
                sel_idx = torch.arange(N, device=self.device)
            else:
                sel_idx = torch.topk(score, k=kk, dim=0, largest=True, sorted=False).indices
            mask_neurons = torch.zeros(N, dtype=out.dtype, device=self.device)
            mask_neurons[sel_idx] = 1.0

        return self._apply_mask(out, mask_neurons, flatten_spatial=self.cfg.flatten_spatial)

    # ---- utils ----
    def _init_buffers(self, N: int) -> None:
        dev = self.device
        self._neurons_N = N
        self._tr = torch.zeros(N, device=dev)
        self._ema = torch.zeros(N, device=dev)
        self._counters = torch.zeros(N, device=dev)
        self._vt_bias = torch.zeros(N, device=dev)
        self._t_seen = 0

    @staticmethod
    def _resolve_k(k: float, N: int) -> int:
        if k < 1.0:
            kk = max(1, int(round(float(k) * N)))
        else:
            kk = int(round(k))
        return min(max(1, kk), N)

    def _reduce_activity(self, out: Tensor, flatten_spatial: bool) -> Tensor:
        if not flatten_spatial:
            dims = (0,) + tuple(range(2, out.dim()))
            return out.mean(dim=dims)
        B = out.shape[0]
        flat = out.view(B, -1)
        return flat.mean(dim=0)

    def _compute_vt_bias(self) -> Tensor:
        vt = torch.clamp(self.cfg.vt_scale * self._ema, min=self.cfg.th_min, max=self.cfg.th_max)
        self._vt_bias = vt
        return vt

    def _apply_mask(self, out: Tensor, mask_neurons: Tensor, flatten_spatial: bool) -> Tensor:
        if not flatten_spatial:
            shape = [1, -1] + [1] * (out.dim() - 2)
            return out * mask_neurons.view(*shape)
        B, C = out.shape[:2]
        spatial = out.shape[2:]
        HW = int(torch.prod(torch.tensor(spatial)).item()) if spatial else 1
        N = C * HW
        if N != mask_neurons.numel():
            return out
        if HW == 1:
            return out * mask_neurons.view(1, C)
        return out * mask_neurons.view(1, C, *spatial)

    # ---- API BaseMethod ----
    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def before_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # reset opcional por tarea
        if self.cfg.reset_counters_each_task and self._counters is not None:
            with torch.no_grad():
                self._counters.zero_()

        # localizar target en ESTE modelo (por si cambia entre runs)
        target = self._find_target_module(model, self.cfg.attach_to)
        if target is None:
            raise ValueError(f"[SA-SNN] No se encontró submódulo '{self.cfg.attach_to}'")
        self._target = target

        # registrar hook sólo si no estaba
        if self._hook_handle is None:
            self._hook_handle = self._target.register_forward_hook(self._forward_hook)

        # Importante: NO llamar a self._target(xb). El hook se activará en el forward normal.

    def after_task(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        pass

    def detach(self) -> None:
        try:
            if self._hook_handle is not None:
                self._hook_handle.remove()
        finally:
            self._hook_handle = None
            self._target = None
            self._tr = None
            self._ema = None
            self._counters = None
            self._vt_bias = None
            self._neurons_N = None
