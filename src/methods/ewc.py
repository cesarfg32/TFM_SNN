# src/methods/ewc.py
# -*- coding: utf-8 -*-
"""EWC (Elastic Weight Consolidation) para mitigar el olvido catastrófico."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from src.nn_io import _forward_with_cached_orientation, _align_target_shape

@dataclass
class EWCConfig:
    lambd: float = 0.0
    fisher_batches: int = 25

class EWC:
    """Implementación EWC con Fisher diagonal."""
    def __init__(self, model: nn.Module, cfg: EWCConfig):
        self.model = model
        self.cfg   = cfg
        self._mu: Dict[str, torch.Tensor]     = {}
        self._fisher: Dict[str, torch.Tensor] = {}

    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def estimate_fisher(self, loader, device: torch.device, loss_fn: nn.Module | None = None) -> int:
        """
        Estima Fisher ≈ E[(∂L/∂θ)^2].
        - Asegura modelo en `device`.
        - ¡Con gradientes habilitados! (sin no_grad)
        """
        device = torch.device(device)
        self.model.to(device)

        was_training = self.model.training
        self.model.eval()  # eval para desactivar dropout/BN updates (pero con grad)

        fisher_loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        # Acumuladores (en el device correcto)
        fisher = {
            n: torch.zeros_like(p, device=p.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        # Congela θ*
        self._mu = self._snapshot_params()

        n_batches = 0
        cap = int(self.cfg.fisher_batches)
        phase_hint: Dict[str, str] = {"fisher": None}

        # Asegura que los gradientes están habilitados (por si alguien activó global no_grad)
        with torch.enable_grad():
            for x, y in loader:
                y = y.to(device, non_blocking=True)

                # forward (AMP-safe dentro de _forward_with_cached_orientation)
                y_hat = _forward_with_cached_orientation(
                    model=self.model, x=x, y=y, device=device, use_amp=True, phase_hint=phase_hint, phase="fisher"
                )
                # loss en FP32 con grafo
                y_aligned = _align_target_shape(y_hat, y).to(device=y_hat.device, dtype=y_hat.dtype, non_blocking=True)
                loss = fisher_loss_fn(y_hat.to(torch.float32), y_aligned.to(torch.float32)).mean()

                # backward
                self.model.zero_grad(set_to_none=True)
                loss.backward()

                # acumula grad^2
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach().pow(2)

                n_batches += 1
                if n_batches >= cap:
                    break

        # Promedia
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
            dev = next(self.model.parameters()).device
            return torch.tensor(0.0, device=dev)
        reg = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher:
                reg = reg + (self._fisher[n] * (p - self._mu[n]).pow(2)).sum()
        return self.cfg.lambd * reg

class EWCMethod:
    name = "ewc"
    def __init__(self, model: nn.Module, lambd: float, fisher_batches: int, loss_fn: nn.Module, device: torch.device):
        self.loss_fn = loss_fn
        self.device = device
        self.impl = EWC(model, EWCConfig(lambd=float(lambd), fisher_batches=int(fisher_batches)))
        self.name = f"ewc_lam_{lambd:.0e}"
        self.inner_verbose = getattr(self, "inner_verbose", False)
        self.inner_every   = getattr(self, "inner_every", 50)

    def penalty(self) -> torch.Tensor:
        return self.impl.penalty()

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        used = self.impl.estimate_fisher(train_loader, device=self.device, loss_fn=self.loss_fn)
        with torch.no_grad():
            if self.impl._fisher:
                abs_sum = 0.0
                abs_max = 0.0
                for f in self.impl._fisher.values():
                    if f is not None:
                        abs_sum += float(f.abs().sum().item())
                        cur_max = float(f.abs().max().item())
                        if cur_max > abs_max:
                            abs_max = cur_max
                print(f"[EWC] Fisher listo: batches_usados={used} | sum={abs_sum:.3e} | max={abs_max:.3e}")
            else:
                print(f"[EWC] Fisher listo: batches_usados={used} | (vacío)")

    def after_task(self, model: nn.Module, train_loader, val_loader) -> None:
        pass

    def tunable(self) -> dict:
        try:
            val = float(self.impl.cfg.lambd)
        except Exception:
            val = 0.0
        return {"param": "lambd", "value": val, "strategy": "ratio", "target_ratio": 1.0}
