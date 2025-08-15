# src/methods/ewc.py
from dataclasses import dataclass
from typing import Dict, Iterable
import torch, torch.nn as nn

@dataclass
class EWCConfig:
    lambd: float = 1e10
    fisher_batches: int = 25

class EWC:
    def __init__(self, model: nn.Module, config: EWCConfig):
        self.model = model
        self.cfg = config
        self.fisher: Dict[str, torch.Tensor] = {}
        self.theta_old: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _save_theta(self):
        self.theta_old = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    @staticmethod
    def _to_model_shape(x: torch.Tensor) -> torch.Tensor:
        """Convierte (B,T,C,H,W) -> (T,B,C,H,W) si aplica."""
        return x.permute(1, 0, 2, 3, 4).contiguous() if x.dim() == 5 else x

    def estimate_fisher(self, loader: Iterable, loss_fn: nn.Module, device: torch.device):
        # Inicializa el diccionario de Fisher
        self.fisher = {
            n: torch.zeros_like(p, device=device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.train(False)
        n_batches = 0

        for i, (x, y) in enumerate(loader):
            if n_batches >= self.cfg.fisher_batches:
                break

            x = x.to(device)
            y = y.to(device)
            x = self._to_model_shape(x)  # <-- ¡Permute aquí!

            self.model.zero_grad(set_to_none=True)
            y_hat = self.model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()

            with torch.no_grad():
                for (n, p) in self.model.named_parameters():
                    if p.grad is None or not p.requires_grad:
                        continue
                    self.fisher[n] += p.grad.detach() ** 2

            n_batches += 1

        if n_batches > 0:
            for n in self.fisher:
                self.fisher[n] /= float(n_batches)

        self._save_theta()

    def penalty(self) -> torch.Tensor:
        if not self.fisher or not self.theta_old:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss_ewc = 0.0
        for (n, p) in self.model.named_parameters():
            if n not in self.fisher or n not in self.theta_old or not p.requires_grad:
                continue
            loss_ewc = loss_ewc + 0.5 * self.cfg.lambd * torch.sum(
                self.fisher[n] * (p - self.theta_old[n]) ** 2
            )
        return loss_ewc
