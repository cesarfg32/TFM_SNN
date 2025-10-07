# -*- coding: utf-8 -*-
"""
EWC (Elastic Weight Consolidation) para mitigar el olvido catastrófico.

Puntos clave:
- estimate_fisher():
  * model.eval() durante la estimación, y restaura el modo previo al final.
  * SIN AMP/autocast para estabilidad numérica (usa torch.amp.autocast(..., enabled=False)).
  * acumula grad^2 a lo largo de 'fisher_batches' y promedia.
  * Invariante a batch size: se fuerza loss.mean() y NO se multiplica por B.
  * Ajuste robusto de formas: si y_hat=(B,1) y y=(B,), se usa y=(B,1).
- penalty(): λ * Σ F_i (θ_i - θ*_i)^2
- EWCMethod.after_task(): trazas de Fisher controlables por env var EWC_FISHER_LOG (1=ON, 0=OFF)
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn, amp as ta


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
    """Hiperparámetros de EWC."""
    lambd: float = 0.0           # intensidad de la penalización
    fisher_batches: int = 25     # nº de batches para estimar el Fisher (media de grad^2)


class EWC:
    """
    Implementación sencilla de EWC con Fisher diagonal.

    - penalty(): devuelve el término λ * Σ F_i (θ_i - θ*_i)^2
    - estimate_fisher(): estima F_i como media de grad^2 en 'fisher_batches' minibatches
    """
    def __init__(self, model: nn.Module, cfg: EWCConfig):
        self.model = model
        self.cfg = cfg
        self._fisher: dict[str, torch.Tensor] = {}
        self._mu: dict[str, torch.Tensor] = {}   # copia de parámetros tras consolidar cada tarea

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

        - Modo eval() para desactivar dropout/bn durante la estimación.
        - SIN autocast/AMP para mayor estabilidad numérica (torch.amp.autocast(..., enabled=False)).
        - Ajuste de formas para pérdidas de regresión (p.ej. y=(B,) vs y_hat=(B,1)).
        - Invariante a batch size: se fuerza loss.mean() y NO se multiplica por B.

        Devuelve el número de batches usados.
        """
        # Pasamos a eval y guardamos el estado previo para restaurarlo al final
        was_training = self.model.training
        self.model.eval()

        # Loss robusta por defecto (si no se provee)
        fisher_loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        # Inicializa tensores del mismo tamaño que cada parámetro
        fisher = {
            n: torch.zeros_like(p, device=p.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self._mu = self._snapshot_params()

        n_batches = 0
        cap = int(self.cfg.fisher_batches)

        # Autocast explícito vía torch.amp.autocast (API nueva)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        for x, y in loader:
            x = _permute_if_needed(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)

            # Necesitamos gradiente → no torch.no_grad() aquí
            self.model.zero_grad(set_to_none=True)

            # ---- FORWARD sin AMP y con loss.mean() para estabilizar escala ----
            with ta.autocast(device_type, enabled=False):
                y_hat = self.model(x)
                # Ajuste de forma robusto: si y_hat=(B,1) y y=(B,), pasamos y->(B,1)
                if y_hat.ndim == 2 and y_hat.shape[1] == 1 and y.ndim == 1:
                    y = y.unsqueeze(1)
                loss = fisher_loss_fn(y_hat, y)
                if loss.dim() > 0:
                    loss = loss.mean()

            # backward "normal" (sin GradScaler)
            loss.backward()

            # Acumular grad^2 en cada parámetro (SIN escalar por B)
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.detach().pow(2)

            n_batches += 1
            if n_batches >= cap:
                break

        # Media sobre los batches procesados
        denom = max(1, n_batches)
        for n in fisher:
            fisher[n] /= denom

        # Guardamos la estimación consolidada
        self._fisher = {n: f.detach().clone() for n, f in fisher.items()}

        # Restaurar modo entrenamiento si estaba así
        if was_training:
            self.model.train()

        return n_batches

    def penalty(self) -> torch.Tensor:
        """
        λ * Σ F_i (θ_i - θ*_i)^2. Si no hay Fisher estimado, devuelve 0 (constante)
        para no romper backward ni sacar '-0.0' en logs.
        """
        if not self._fisher:
            # Constante 0.0 en el device correcto; no hace falta "enganchar" el grad.
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
