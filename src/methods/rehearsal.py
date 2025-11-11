# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator

import random
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .base import BaseMethod


@dataclass
class RehearsalConfig:
    buffer_size: int = 10_000
    replay_ratio: float = 0.2


class ReservoirBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = int(buffer_size)
        self.reservoir: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_seen: int = 0

    def add_sample(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = sample
        x = x.detach().cpu().clone()
        y = y.detach().cpu().clone()
        self.num_seen += 1
        if len(self.reservoir) < self.buffer_size:
            self.reservoir.append((x, y))
        else:
            j = random.randint(0, self.num_seen - 1)
            if j < self.buffer_size:
                self.reservoir[j] = (x, y)

    def add_from_loader(self, loader: DataLoader, cap: Optional[int] = None) -> int:
        added = 0
        for xb, yb in loader:
            B = xb.shape[0]
            for b in range(B):
                self.add_sample((xb[b], yb[b]))
                added += 1
                if cap is not None and added >= cap:
                    return added
        return added

    def as_dataset(self) -> Dataset:
        if not self.reservoir:
            raise RuntimeError("Reservoir vacío")
        xs, ys = zip(*self.reservoir)
        X = torch.stack(xs, dim=0)
        Y = torch.stack(ys, dim=0)
        return TensorDataset(X, Y)

    def __len__(self) -> int:
        return len(self.reservoir)


class _ReplayMixLoader:
    def __init__(self, task_loader: DataLoader, replay_loader: DataLoader, task_bs: int, replay_bs: int):
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
        x = torch.cat([x_t, x_r], dim=0)
        y = torch.cat([y_t, y_r], dim=0)
        return x, y

    def __len__(self):
        return len(self.task_loader)


class RehearsalMethod(BaseMethod):
    """Uniforme: acepta buffer_size y replay_ratio directos en __init__ + device/loss_fn."""
    name = "rehearsal"

    def __init__(
        self,
        *,
        buffer_size: int = 10_000,
        replay_ratio: float = 0.2,
        device: Optional[torch.device] = None,
        loss_fn=None,
    ):
        super().__init__(device=device, loss_fn=loss_fn)
        assert 0.0 <= float(replay_ratio) <= 1.0, "replay_ratio debe estar en [0,1]"
        self.cfg = RehearsalConfig(buffer_size=int(buffer_size), replay_ratio=float(replay_ratio))
        self.buffer = ReservoirBuffer(self.cfg.buffer_size)
        self._last_task_loader: Optional[DataLoader] = None
        rr = int(round(self.cfg.replay_ratio * 100))
        self.name = f"rehearsal_buf_{self.cfg.buffer_size}_rr_{rr}"

        # flags de logging (opcionales)
        self.buffer_verbose: bool = False
        self.replay_verbose: bool = False
        self.replay_show_shapes: bool = False

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def before_task(self, model, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self._last_task_loader = train_loader

    def after_task(self, model, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if self._last_task_loader is not None:
            before = len(self.buffer)
            added = self.buffer.add_from_loader(self._last_task_loader)
            now = len(self.buffer)
            self._last_task_loader = None
            if getattr(self, "buffer_verbose", False):
                cap = self.buffer.buffer_size
                print(f"[Rehearsal] buffer += {added} (total={now}/{cap}) | seen={self.buffer.num_seen}")

    def prepare_train_loader(self, train_loader: DataLoader):
        if len(self.buffer) == 0 or self.cfg.replay_ratio <= 0.0:
            return train_loader

        total_bs = int(getattr(train_loader, "batch_size", None) or 0)
        if total_bs <= 0:
            try:
                xb, _ = next(iter(train_loader))
                total_bs = int(xb.shape[0])
            except Exception:
                total_bs = 8

        rep_bs = max(1, int(round(total_bs * self.cfg.replay_ratio)))
        task_bs = max(1, total_bs - rep_bs)
        if task_bs + rep_bs != total_bs:
            rep_bs = total_bs - task_bs

        if getattr(self, "replay_verbose", False):
            msg = f"[Rehearsal] mix: task_bs={task_bs} | replay_bs={rep_bs} | total_bs={total_bs} | buf_len={len(self.buffer)}"
            if getattr(self, "replay_show_shapes", False):
                try:
                    xb_t, _ = next(iter(train_loader))
                    xb_r, _ = next(iter(DataLoader(self.buffer.as_dataset(), batch_size=rep_bs)))
                    msg += f" | x_task.shape={tuple(xb_t.shape)} x_rep.shape={tuple(xb_r.shape)}"
                except Exception:
                    pass
            print(msg)

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

    def detach(self) -> None:
        self.buffer.reservoir.clear()
