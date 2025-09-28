# src/methods/ewc.py
# -*- coding: utf-8 -*-
"""
EWC (Elastic Weight Consolidation) para mitigar el olvido catastrófico.

Cambios clave:
- estimate_fisher() ahora pone el modelo en eval() durante la estimación y
  vuelve a train() al final (dropout/bn desactivados mientras se estima).
- Se asegura la permutación (B,T,C,H,W) -> (T,B,C,H,W) antes del forward,
  igual que en training.py.
- No usa AMP en la estimación del Fisher (evitamos ruido numérico).
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

# Importamos el helper de permutación desde training para no duplicar lógica
try:
    from src.training import _permute_if_needed
except Exception:
    # Fallback por si se usa este módulo de forma aislada
    def _permute_if_needed(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5 and x.shape[0] != x.shape[1]:
            return x.permute(1, 0, 2, 3, 4).contiguous()
        return x


@dataclass
class EWCConfig:
    """Hiperaparámetros de EWC."""
    lambd: float = 0.0           # intensidad de la penalización
    fisher_batches: int = 25     # nº de batches para estimar el Fisher (media de grad^2)


class EWC:
    """
    Implementación sencilla de EWC con Fisher diagonal.
    - penalty(): devuelve el término de regularización λ * Σ F_i (θ_i - θ*_i)^2
    - estimate_fisher(): estima F_i como la media del gradiente al cuadrado en 'fisher_batches' minibatches
    """

    def __init__(self, model: nn.Module, cfg: EWCConfig):
        self.model = model
        self.cfg = cfg
        self._fisher: dict[str, torch.Tensor] = {}
        self._mu: dict[str, torch.Tensor] = {}  # copia de parámetros tras consolidar cada tarea

    @torch.no_grad()
    def _snapshot_params(self) -> dict[str, torch.Tensor]:
        """Copia de los parámetros actuales (θ*)."""
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def estimate_fisher(self, loader, loss_fn: nn.Module, device: torch.device) -> None:
        """
        Estima el Fisher diagonal promediando grad^2 en 'fisher_batches' minibatches del loader dado.
        - Modo eval() para desactivar dropout/bn durante la estimación.
        - Sin autocast/AMP para mayor estabilidad numérica en la medida.
        """
        # Pasamos a eval y guardamos el estado previo para restaurarlo al final
        was_training = self.model.training
        self.model.eval()

        # Inicializa tensores del mismo tamaño que cada parámetro
        fisher = {n: torch.zeros_like(p, device=p.device) for n, p in self.model.named_parameters()
                  if p.requires_grad}
        self._mu = self._snapshot_params()

        n_batches = 0
        for x, y in loader:
            x = _permute_if_needed(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)

            # Necesitamos grad para acumular grad^2 → sin torch.no_grad()
            self.model.zero_grad(set_to_none=True)
            y_hat = self.model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()  # acumula gradientes

            # Acumular grad^2 en cada parámetro
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach().pow(2)

            n_batches += 1
            if n_batches >= int(self.cfg.fisher_batches):
                break

        # Media
        denom = max(1, n_batches)
        for n in fisher:
            fisher[n] /= denom

        # Guardamos la estimación consolidada
        self._fisher = {n: f.detach().clone() for n, f in fisher.items()}

        # Restaurar modo entrenamiento si estaba así
        if was_training:
            self.model.train()

    def penalty(self) -> torch.Tensor:
        """λ * Σ F_i (θ_i - θ*_i)^2. Si no hay Fisher estimado, devuelve 0."""
        if not self._fisher:
            # Devolvemos un 0 en el mismo device, con grad para no romper backward
            return next(self.model.parameters()).sum() * 0.0

        reg = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher:
                reg = reg + (self._fisher[n] * (p - self._mu[n]).pow(2)).sum()

        return self.cfg.lambd * reg

# --- Wrapper para la API de métodos (sin tocar tu EWC original) ---

class EWCMethod:
    name = "ewc"

    def __init__(self, model: nn.Module, lambd: float, fisher_batches: int,
                 loss_fn: nn.Module, device: torch.device):
        # Reutiliza tu implementación exactamente igual
        self.loss_fn = loss_fn
        self.device = device
        self.impl = EWC(model, EWCConfig(lambd=float(lambd), fisher_batches=int(fisher_batches)))
        self.name = f"ewc_lam_{lambd:.0e}"

    def penalty(self) -> torch.Tensor:
        return self.impl.penalty()

    def before_task(self, model: nn.Module, train_loader, val_loader) -> None:
        # no-op
        pass

    def after_task(self, model, train_loader, val_loader) -> None:
        # Preferimos val si aporta al menos fisher_batches; si no, usamos train
        fisher_batches = getattr(self.impl.cfg, "fisher_batches", 100)
        try:
            chosen = val_loader if len(val_loader) >= fisher_batches else train_loader
        except TypeError:
            # Si el val_loader no tiene len(), usamos train
            chosen = train_loader

        self.impl.estimate_fisher(chosen, self.loss_fn, device=self.device)