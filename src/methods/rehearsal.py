# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator

import random
import torch
from torch.utils.data import DataLoader, Dataset

from .base import BaseMethod


@dataclass
class RehearsalConfig:
    buffer_size: int = 10_000
    replay_ratio: float = 0.2
    compress_mode: str = "auto"   # "auto" | "u8" | "fp16" | "none"
    pin_memory: bool = True
    max_total_bs: Optional[int] = None  # ← NUEVO: tope al batch efectivo (task+replay)


class CompressedReservoirBuffer:
    """
    Reservoir con compresión:
      - "u8": spikes binarios 0/1
      - "fp16": entradas continuas
      - "none": float32
      - "auto": detecta binariedad
    Targets `y` guardados como escalares 0-D; el DataLoader los apila a (B,).
    """
    def __init__(self, buffer_size: int, compress_mode: str = "auto", pin_memory: bool = True):
        self.buffer_size = int(buffer_size)
        self.mode = str(compress_mode).lower()
        self.pin = bool(pin_memory)
        self.reservoir: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_seen: int = 0
        self._shape: Optional[Tuple[int, ...]] = None

    @staticmethod
    def _is_binary01(x: torch.Tensor) -> bool:
        if not x.dtype.is_floating_point:
            return False
        xmin = x.min().item()
        xmax = x.max().item()
        if xmin < 0.0 or xmax > 1.0:
            return False
        return torch.all((x == 0) | (x == 1)).item()

    def _decide_mode(self, x: torch.Tensor) -> str:
        if self.mode in {"u8", "fp16", "none"}:
            return self.mode
        try:
            return "u8" if self._is_binary01(x) else "fp16"
        except Exception:
            return "fp16"

    def add_sample(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = sample
        x = (x.detach().cpu() if x.is_cuda else x.detach()).contiguous()
        y = (y.detach().cpu() if y.is_cuda else y.detach()).contiguous()
        y = y.to(dtype=torch.float32).reshape(())  # escalar 0-D

        if self._shape is None:
            self._shape = tuple(x.shape)

        mode = self._decide_mode(x)
        if mode == "u8":
            x_comp = x.to(dtype=torch.uint8, copy=True)
        elif mode == "fp16":
            x_comp = x.to(dtype=torch.float16, copy=True)
        else:
            x_comp = x.to(dtype=torch.float32, copy=True)

        if self.pin and torch.cuda.is_available():
            try:
                x_comp = x_comp.pin_memory()
                y = y.pin_memory()
            except Exception:
                pass

        self.num_seen += 1
        if len(self.reservoir) < self.buffer_size:
            self.reservoir.append((x_comp, y))
        else:
            j = random.randint(0, self.num_seen - 1)
            if j < self.buffer_size:
                self.reservoir[j] = (x_comp, y)

    def add_from_loader(self, loader: DataLoader, cap: Optional[int] = None) -> int:
        added = 0
        for xb, yb in loader:
            B = int(xb.shape[0])
            for b in range(B):
                self.add_sample((xb[b], yb[b]))
                added += 1
                if cap is not None and added >= cap:
                    return added
        return added

    def __len__(self) -> int:
        return len(self.reservoir)

    class _ReplayDataset(Dataset):
        def __init__(self, parent: "CompressedReservoirBuffer"):
            self.parent = parent

        def __len__(self) -> int:
            return len(self.parent.reservoir)

        def __getitem__(self, idx: int):
            x_comp, y = self.parent.reservoir[idx]
            if x_comp.dtype in (torch.uint8, torch.int8):
                x = x_comp.float()
            elif x_comp.dtype == torch.float16:
                x = x_comp.float()
            else:
                x = x_comp
            return x.contiguous(), y.to(dtype=torch.float32)

    def as_dataset(self) -> Dataset:
        if not self.reservoir:
            raise RuntimeError("Reservoir vacío")
        return CompressedReservoirBuffer._ReplayDataset(self)


class _ReplayMixLoader:
    _is_replay_mix = True

    def __init__(self, task_loader: DataLoader, replay_loader: DataLoader, task_bs: int, replay_bs: int):
        self.task_loader = task_loader
        self.replay_loader = replay_loader
        self.task_bs = int(task_bs)
        self.replay_bs = int(replay_bs)
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

        # Recorte para respetar el tope efectivo (si task/replay loaders trajeran más)
        if self.task_bs < x_t.shape[0]:
            x_t = x_t[: self.task_bs]
            y_t = y_t[: self.task_bs]
        if self.replay_bs < x_r.shape[0]:
            x_r = x_r[: self.replay_bs]
            y_r = y_r[: self.replay_bs]

        if y_t.ndim != 1:
            y_t = y_t.view(-1)
        if y_r.ndim != 1:
            y_r = y_r.view(-1)

        x = torch.cat([x_t, x_r], dim=0)
        y = torch.cat([y_t, y_r], dim=0)
        return x.contiguous(), y.contiguous()

    def __len__(self):
        return len(self.task_loader)


class RehearsalMethod(BaseMethod):
    name = "rehearsal"

    def __init__(
        self,
        *,
        buffer_size: int = 10_000,
        replay_ratio: float = 0.2,
        compress_mode: str = "auto",
        pin_memory: bool = True,
        max_total_bs: Optional[int] = None,  # ← NUEVO
        device: Optional[torch.device] = None,
        loss_fn=None,
    ):
        super().__init__(device=device, loss_fn=loss_fn)
        assert 0.0 <= float(replay_ratio) <= 1.0, "replay_ratio debe estar en [0,1]"

        self.cfg = RehearsalConfig(
            buffer_size=int(buffer_size),
            replay_ratio=float(replay_ratio),
            compress_mode=str(compress_mode).lower(),
            pin_memory=bool(pin_memory),
            max_total_bs=(int(max_total_bs) if max_total_bs is not None else None),
        )
        self.buffer = CompressedReservoirBuffer(
            self.cfg.buffer_size,
            compress_mode=self.cfg.compress_mode,
            pin_memory=self.cfg.pin_memory,
        )

        self._last_task_loader: Optional[DataLoader] = None
        self._mix_loader: Optional[_ReplayMixLoader] = None

        rr = int(round(self.cfg.replay_ratio * 100))
        self.name = f"rehearsal_buf_{self.cfg.buffer_size}_rr_{rr}"

        self.buffer_verbose: bool = False
        self.replay_verbose: bool = False
        self.replay_show_shapes: bool = False

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def before_task(self, model, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self._last_task_loader = train_loader
        self._mix_loader = None

    def after_task(self, model, train_loader: DataLoader, val_loader: DataLoader) -> None:
        if self._last_task_loader is not None:
            added = self.buffer.add_from_loader(self._last_task_loader)
            now = len(self.buffer)
            self._last_task_loader = None
            self._mix_loader = None
            if getattr(self, "buffer_verbose", False):
                cap = self.buffer.buffer_size
                print(f"[Rehearsal] buffer += {added} (total={now}/{cap}) | seen={self.buffer.num_seen}")

    def prepare_train_loader(self, train_loader: DataLoader):
        if self._mix_loader is not None:
            return self._mix_loader
        if hasattr(train_loader, "_is_replay_mix"):
            return train_loader
        if len(self.buffer) == 0 or self.cfg.replay_ratio <= 0.0:
            return train_loader

        total_bs = int(getattr(train_loader, "batch_size", None) or 0)
        if total_bs <= 0:
            try:
                xb, _ = next(iter(train_loader))
                total_bs = int(xb.shape[0])
            except Exception:
                total_bs = 8

        # Tope efectivo si se ha configurado
        target_total = total_bs
        if self.cfg.max_total_bs is not None:
            target_total = max(2, min(total_bs, int(self.cfg.max_total_bs)))

        rep_bs = max(1, int(round(target_total * self.cfg.replay_ratio)))
        task_bs = max(1, target_total - rep_bs)
        if task_bs + rep_bs != target_total:
            rep_bs = target_total - task_bs

        if getattr(self, "replay_verbose", False):
            msg = (
                f"[Rehearsal] mix: task_bs={task_bs} | replay_bs={rep_bs} | "
                f"total_bs={target_total} (base={total_bs}) | buf_len={len(self.buffer)}"
            )
            if getattr(self, "replay_show_shapes", False):
                try:
                    xb_t, _ = next(iter(train_loader))
                    from torch.utils.data import DataLoader as _DL
                    xb_r, _ = next(iter(_DL(self.buffer.as_dataset(), batch_size=rep_bs, shuffle=True)))
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

        self._mix_loader = _ReplayMixLoader(train_loader, replay_loader, task_bs, rep_bs)
        return self._mix_loader

    def detach(self) -> None:
        self.buffer.reservoir.clear()
        self._mix_loader = None
