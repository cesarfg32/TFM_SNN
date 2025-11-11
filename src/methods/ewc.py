# -*- coding: utf-8 -*-
"""EWC (Elastic Weight Consolidation) para mitigar el olvido catastrófico."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from src.methods.base import BaseMethod
from src.nn_io import _forward_with_cached_orientation, _align_target_shape


@dataclass
class EWCConfig:
    lambd: float = 0.0
    fisher_batches: int = 25


class EWC:
    """Implementación EWC con Fisher diagonal (modelo perezoso, se inyecta en before_task)."""
    name = "ewc"

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        cfg: EWCConfig,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.loss_fn = loss_fn or nn.MSELoss()
        self.device = (
            device
            if device is not None
            else (torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
        )
        self._mu: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}

    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        assert self.model is not None, "[EWC] model no inicializado"
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def estimate_fisher(self, loader) -> int:
        """Estima Fisher ≈ E[(∂L/∂θ)^2] usando batches del loader."""
        assert self.model is not None, "[EWC] model no inicializado"
        self.model.to(self.device)
        was_training = self.model.training
        self.model.eval()

        fisher = {n: torch.zeros_like(p, device=p.device)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self._mu = self._snapshot_params()

        n_batches = 0
        cap = int(self.cfg.fisher_batches)
        hint = {"fisher": None}

        with torch.enable_grad():
            for x, y in loader:
                y = y.to(self.device, non_blocking=True)
                y_hat = _forward_with_cached_orientation(
                    self.model, x, y, self.device, use_amp=True, phase_hint=hint, phase="fisher"
                )
                y_aligned = _align_target_shape(y_hat, y).to(
                    device=y_hat.device, dtype=y_hat.dtype, non_blocking=True
                )
                loss = self.loss_fn(y_hat.to(torch.float32), y_aligned.to(torch.float32)).mean()

                self.model.zero_grad(set_to_none=True)
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach().pow(2)
                n_batches += 1
                if n_batches >= cap:
                    break

        if n_batches > 0:
            for n in fisher:
                fisher[n] /= float(n_batches)

        self._fisher = fisher
        if was_training:
            self.model.train()
        return n_batches

    def penalty(self) -> torch.Tensor:
        """λ * Σ F_i (θ_i - θ*_i)^2."""
        if not self._fisher:
            return torch.tensor(0.0, device=self.device)
        assert self.model is not None, "[EWC] model no inicializado"
        reg = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher:
                reg = reg + (self._fisher[n] * (p - self._mu[n]).pow(2)).sum()
        return float(self.cfg.lambd) * reg


class EWCMethod(BaseMethod):
    """Método EWC homogéneo al resto: no requiere model en __init__, device/loss_fn se pasan aquí."""
    name = "ewc"

    def __init__(
        self,
        *,
        lambd: float,
        fisher_batches: int = 100,
        device: Optional[torch.device] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__(device=device, loss_fn=loss_fn)
        self.impl = EWC(
            model=None,
            cfg=EWCConfig(lambd=float(lambd), fisher_batches=int(fisher_batches)),
            loss_fn=loss_fn,
            device=self.device,
        )
        self.name = f"ewc_lam_{lambd:.0e}"
        self.inner_verbose = getattr(self, "inner_verbose", False)
        self.inner_every = getattr(self, "inner_every", 50)

    def penalty(self) -> torch.Tensor:
        return self.impl.penalty()

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        # Inyecta el modelo y (si procede) alinea el device con el del modelo
        self.impl.model = model
        try:
            dev_model = next(model.parameters()).device
            if self.impl.device != dev_model:
                self.impl.device = dev_model
        except StopIteration:
            pass

        used = self.impl.estimate_fisher(train_loader)
        try:
            if self.impl._fisher:
                abs_sum = float(sum(f.abs().sum().item() for f in self.impl._fisher.values()))
                abs_max = float(max(f.abs().max().item() for f in self.impl._fisher.values()))
                print(f"[EWC] Fisher listo: batches_usados={used} | sum={abs_sum:.3e} | max={abs_max:.3e}")
            else:
                print(f"[EWC] Fisher listo: batches_usados={used} | (vacío)")
        except Exception:
            pass

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        pass

    def tunable(self) -> dict:
        try:
            val = float(self.impl.cfg.lambd)
        except Exception:
            val = 0.0
        return {"param": "lambd", "value": val, "strategy": "ratio", "target_ratio": 1.0}
