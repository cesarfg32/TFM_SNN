# src/methods/rehearsal.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator
import random

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .api import ContinualMethod


@dataclass
class RehearsalConfig:
    buffer_size: int = 10_000
    replay_ratio: float = 0.2  # fracción del batch dedicada a replay en cada iteración


class ReservoirBuffer:
    """Reservoir sampling en CPU: guarda (x,y) como tensores."""
    def __init__(self, buffer_size: int):
        self.buffer_size = int(buffer_size)
        self.reservoir: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_seen: int = 0

    def add_sample(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = sample
        # guardamos siempre en CPU para no ocupar VRAM; clonado para evitar referencias
        x = x.detach().cpu().clone()
        y = y.detach().cpu().clone()
        self.num_seen += 1
        if len(self.reservoir) < self.buffer_size:
            self.reservoir.append((x, y))
        else:
            j = random.randint(0, self.num_seen - 1)
            if j < self.buffer_size:
                self.reservoir[j] = (x, y)

    def add_from_loader(self, loader: DataLoader, cap: Optional[int] = None) -> None:
        """Añade muestras individuales desde un loader (máximo 'cap' muestras si se desea)."""
        added = 0
        for xb, yb in loader:
            # xb puede ser (B, T, C, H, W) o (B, C, H, W) según encoder del dataset
            B = xb.shape[0]
            for b in range(B):
                self.add_sample((xb[b], yb[b]))
                added += 1
                if cap is not None and added >= cap:
                    return

    def as_dataset(self) -> Dataset:
        if not self.reservoir:
            raise RuntimeError("Reservoir vacío")
        xs, ys = zip(*self.reservoir)
        X = torch.stack(xs, dim=0)  # se apila (mantiene 4D o 5D por muestra)
        Y = torch.stack(ys, dim=0)
        return TensorDataset(X, Y)

    def __len__(self) -> int:
        return len(self.reservoir)


class _ReplayMixLoader:
    """Iterador que mezcla batchs de tarea y replay, sin mover a device (lo hace training)."""
    def __init__(self, task_loader: DataLoader, replay_loader: DataLoader,
                 task_bs: int, replay_bs: int):
        self.task_loader = task_loader
        self.replay_loader = replay_loader
        self.task_bs = task_bs
        self.replay_bs = replay_bs

        self._task_it: Optional[Iterator] = None
        self._rep_it: Optional[Iterator] = None

    def __iter__(self):
        self._task_it = iter(self.task_loader)
        self._rep_it = iter(self.replay_loader)
        return self

    def __next__(self):
        if self._task_it is None or self._rep_it is None:
            self.__iter__()
        try:
            x_t, y_t = next(self._task_it)
        except StopIteration:
            raise StopIteration
        try:
            x_r, y_r = next(self._rep_it)
        except StopIteration:
            self._rep_it = iter(self.replay_loader)
            x_r, y_r = next(self._rep_it)

        # Concatenación por batch (dim=0). Formas compatibles: (B, T, C, H, W) o (B, C, H, W)
        x = torch.cat([x_t, x_r], dim=0)
        y = torch.cat([y_t, y_r], dim=0)
        return x, y

    def __len__(self):
        return len(self.task_loader)

class RehearsalMethod:
    def __init__(self, cfg: RehearsalConfig):
        assert 0.0 <= cfg.replay_ratio <= 1.0, "replay_ratio debe estar en [0,1]"
        self.cfg = cfg
        self.buffer = ReservoirBuffer(cfg.buffer_size)
        self._last_task_loader: Optional[DataLoader] = None

        # nombre autoexplicativo p/outputs
        rr = int(round(cfg.replay_ratio * 100))
        self.name = f"rehearsal_buf_{cfg.buffer_size}_rr_{rr}"

    # --- API ContinualMethod ---
    def penalty(self) -> torch.Tensor:
        # No añade regularización de pérdida
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.zeros((), dtype=torch.float32, device=dev)

    def before_task(self, model, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Conserva el loader puro para alimentar el buffer en after_task
        self._last_task_loader = train_loader

    def after_task(self, model, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Al finalizar, añadimos muestras de la tarea al buffer
        if self._last_task_loader is not None:
            self.buffer.add_from_loader(self._last_task_loader)
            self._last_task_loader = None

    # --- Hook opcional para training: envolver el train_loader ---
    def prepare_train_loader(self, train_loader: DataLoader) -> DataLoader | _ReplayMixLoader:
        """Devuelve un loader mixto (tarea+replay) si hay buffer; si no, el original."""
        if len(self.buffer) == 0 or self.cfg.replay_ratio <= 0.0:
            return train_loader

        # tamaños de sub-batch
        total_bs = int(getattr(train_loader, "batch_size", None) or 0)
        if total_bs <= 0:
            # fallback: mira un batch
            try:
                xb, _ = next(iter(train_loader))
                total_bs = int(xb.shape[0])
            except Exception:
                total_bs = 8  # último recurso

        rep_bs = max(1, int(round(total_bs * self.cfg.replay_ratio)))
        task_bs = max(1, total_bs - rep_bs)
        if task_bs + rep_bs != total_bs:
            # ajusta por redondeos
            rep_bs = total_bs - task_bs

        # DataLoader del buffer
        replay_ds = self.buffer.as_dataset()
        replay_loader = DataLoader(
            replay_ds,
            batch_size=rep_bs,
            shuffle=True,
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
            persistent_workers=getattr(train_loader, "persistent_workers", False),
        )
        return _ReplayMixLoader(train_loader, replay_loader, task_bs, rep_bs)
