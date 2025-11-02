# -*- coding: utf-8 -*-
"""EWC (Elastic Weight Consolidation) para mitigar el olvido catastrófico.
Decisiones:
- EWC usa el MISMO helper de training/eval (_forward_with_cached_orientation)
  → unifica orientación 5D y runtime-encode (4D→5D) sin duplicar lógica.
- estimate_fisher():
  * model.eval() durante la estimación, y restaura el modo previo al final.
  * Forward con AMP ACTIVADO (use_amp=True) para replicar training/val y reducir memoria.
  * Acumula grad^2 a lo largo de 'fisher_batches' y promedia.
  * Invariante a batch size: se fuerza loss.mean() y NO se multiplica por B.
  * Alinea y contra y_hat (B vs B×1) de forma robusta y calcula la loss en FP32.
- penalty(): λ * Σ F_i (θ_i - θ*_i)^2
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

# Reutilizamos el helper común (misma ruta que training/eval)
from src.training import _forward_with_cached_orientation

def _align_target_shape(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Alinea forma del target con la predicción:
    - Si y_hat=(B,1) y y=(B,) -> y=(B,1)
    - Si y_hat=(B,)  y y=(B,1) -> y=(B,)
    """
    if y_hat.ndim == 2 and y_hat.shape[1] == 1 and y.ndim == 1:
        return y.unsqueeze(1)
    if y_hat.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1)
    return y


@dataclass
class EWCConfig:
    """Hiperparámetros de EWC."""
    lambd: float = 0.0           # intensidad de la penalización
    fisher_batches: int = 25     # nº de batches para estimar el Fisher (media de grad^2)


class EWC:
    """Implementación sencilla de EWC con Fisher diagonal."""
    def __init__(self, model: nn.Module, cfg: EWCConfig):
        self.model = model
        self.cfg = cfg
        self._fisher: dict[str, torch.Tensor] = {}
        self._mu: dict[str, torch.Tensor] = {}  # copia de parámetros tras consolidar cada tarea

    @torch.no_grad()
    def _snapshot_params(self) -> dict[str, torch.Tensor]:
        """Copia de los parámetros actuales (θ*)."""
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def estimate_fisher(
        self,
        loader,
        loss_fn: nn.Module | None,
        device: torch.device,
    ) -> int:
        """
        Estima el Fisher diagonal promediando grad^2 en 'fisher_batches' minibatches del loader dado.

        - Usa el MISMO helper que training/eval (orientación y runtime-encode).
        - model.eval() durante la estimación; restaura el modo previo al final.
        - Forward con AMP ACTIVADO (use_amp=True) para memoria/estabilidad.
        - Loss en FP32 y con mean() para invariancia a batch size.
        """
        was_training = self.model.training
        self.model.eval()

        fisher_loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        # Inicializa acumuladores
        fisher = {
            n: torch.zeros_like(p, device=p.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        # Congela θ* (después de la tarea)
        self._mu = self._snapshot_params()

        n_batches = 0
        cap = int(self.cfg.fisher_batches)

        # Hint local para la orientación durante Fisher
        phase_hint: Dict[str, str] = {"fisher": None}

        for x, y in loader:
            # y al device (el helper usa B=y.shape[0] para decidir orientación)
            y = y.to(device, non_blocking=True)

            # Forward consistente con training/eval (incluye runtime-encode si aplica)
            # *** CAMBIO 1: use_amp=True ***
            y_hat = _forward_with_cached_orientation(
                model=self.model,
                x=x,
                y=y,
                device=device,
                use_amp=True,
                phase_hint=phase_hint,
                phase="fisher",
            )

            # Alinear shapes/dtypes/devices
            y_aligned = _align_target_shape(y_hat, y).to(
                device=y_hat.device, dtype=y_hat.dtype, non_blocking=True
            )

            # *** CAMBIO 2: loss en FP32 (estable) ***
            y_hat_fp32 = y_hat.to(torch.float32)
            y_fp32 = y_aligned.to(torch.float32)
            loss = fisher_loss_fn(y_hat_fp32, y_fp32)
            if loss.dim() > 0:
                loss = loss.mean()

            # Gradientes (sin GradScaler; ya estamos en FP32 para la loss)
            self.model.zero_grad(set_to_none=True)
            loss.backward()

            # Acumular grad^2
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach().pow(2)

            n_batches += 1
            if n_batches >= cap:
                break

            # Limpieza ligera para evitar fragmentación en secuencias largas
            if torch.cuda.is_available() and (n_batches % 50 == 0):
                torch.cuda.empty_cache()

        # Media sobre los batches procesados
        denom = max(1, n_batches)
        for n in fisher:
            fisher[n] /= denom

        # Guardamos la estimación consolidada
        self._fisher = {n: f.detach().clone() for n, f in fisher.items()}

        # Restaurar modo anterior
        if was_training:
            self.model.train()
        return n_batches

    def penalty(self) -> torch.Tensor:
        """λ * Σ F_i (θ_i - θ*_i)^2. Si no hay Fisher estimado, devuelve 0."""
        if not self._fisher:
            # Constante 0.0 en el device correcto
            dev = next(self.model.parameters()).device
            return torch.tensor(0.0, device=dev)

        reg = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self._fisher:
                reg = reg + (self._fisher[n] * (p - self._mu[n]).pow(2)).sum()
        return self.cfg.lambd * reg


# --- Wrapper para la API de métodos ---
class EWCMethod:
    name = "ewc"

    def __init__(
        self,
        model: nn.Module,
        lambd: float,
        fisher_batches: int,
        loss_fn: nn.Module,
        device: torch.device,
    ):
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
        # Estima Fisher con TRAIN por defecto (suele ser más largo/estable que VAL).
        chosen = train_loader
        try:
            total_len = len(chosen)
        except Exception:
            total_len = None

        if getattr(self, "fisher_verbose", False):
            cap = getattr(self.impl.cfg, "fisher_batches", None)
            if total_len is not None and cap is not None:
                print(f"[EWC] after_task: estimando Fisher en TRAIN (len={total_len}), cap={cap}...")
            else:
                print("[EWC] after_task: estimando Fisher en TRAIN...")

        used = self.impl.estimate_fisher(chosen, self.loss_fn, device=self.device)

        if getattr(self, "fisher_verbose", False):
            # Resumen rápido del Fisher para trazas
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
