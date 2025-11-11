# src/utils.py
# -*- coding: utf-8 -*-
"""
Utilidades comunes del proyecto TFM_SNN (versión limpia, sin back-compat).
- Reproducibilidad (seed_worker)
- Collate para H5 (collate_h5)
- Sampler balanceado por bins (steering)
- DataLoaders desde CSV (runtime encode) u H5 (offline)
- Selector de H5 por atributos y fábrica de loaders por preset

NOTA: No exporta load_preset ni builders de componentes; eso vive en src.utils_components.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import inspect
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import UdacityCSV, ImageTransform, AugmentConfig, H5SpikesDataset


# ---------------------------------------------------------------------
# Reproducibilidad por worker (DataLoader)
# ---------------------------------------------------------------------
def seed_worker(worker_id: int) -> None:
    """Inicializa RNGs por worker para DataLoader multiproceso."""
    import numpy as _np
    worker_seed = torch.initial_seed() % 2**32
    _np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# Collate para H5 (spikes offline)
# ---------------------------------------------------------------------
def collate_h5(batch):
    """
    Collate para tensores H5 (spikes) -> (xb, yb) apilados.
    Espera pares (x, y).
    """
    xs, ys = zip(*batch)
    xb = torch.stack(xs, dim=0).contiguous()
    yb = torch.as_tensor(ys).float()
    return xb, yb


# ---------------------------------------------------------------------
# Sampler balanceado por bins para entrenamiento (reduce sesgo a rectas)
# ---------------------------------------------------------------------
def _make_balanced_sampler(labels: List[float],
                           bins: int = 50,
                           smooth_eps: float = 1e-3) -> WeightedRandomSampler:
    y = np.asarray(labels, dtype=np.float32)
    lo = min(-1.0, float(y.min()))
    hi = max( 1.0, float(y.max()))
    edges = np.linspace(lo, hi, int(bins))
    bin_ids = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)

    counts = np.bincount(bin_ids, minlength=len(edges) - 1).astype(np.float32)
    counts = counts + float(smooth_eps)
    inv = 1.0 / counts
    w = inv[bin_ids]
    weights = torch.as_tensor(w, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)


# ---------------------------------------------------------------------
# DataLoaders CSV (con augment opcional y balanceo) y H5
# ---------------------------------------------------------------------
def make_loaders_from_csvs(
    *,
    base_dir: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    batch_size: int,
    encoder: str,
    T: int,
    gain: float,
    tfm: ImageTransform,
    seed: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = 2,
    aug_train: Optional[AugmentConfig] = None,
    balance_train: bool = False,
    balance_bins: int = 50,
    balance_smooth_eps: float = 1e-3,
    pin_memory_device: str = "cuda",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Crea DataLoaders desde CSV (imágenes) con o sin runtime encode."""
    # Datasets
    tr_ds = UdacityCSV(train_csv, base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm, aug=aug_train)
    va_ds = UdacityCSV(val_csv,   base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm, aug=None)
    te_ds = UdacityCSV(test_csv,  base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm, aug=None)

    # Siembra opcional
    generator = None
    worker_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_fn = seed_worker

    # Balanceo solo en train: desactiva shuffle si se usa sampler
    train_sampler = None
    effective_shuffle = bool(shuffle_train)
    if balance_train:
        train_sampler = _make_balanced_sampler(
            tr_ds.labels, bins=balance_bins, smooth_eps=balance_smooth_eps
        )
        effective_shuffle = False

    # Normaliza workers
    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None
    else:
        if prefetch_factor is None:
            prefetch_factor = 2

    # pin_memory_device si DataLoader lo soporta
    _HAS_PIN_DEVICE = "pin_memory_device" in inspect.signature(DataLoader).parameters
    _dl_extra: Dict[str, Any] = {}
    if pin_memory and _HAS_PIN_DEVICE and pin_memory_device:
        _dl_extra["pin_memory_device"] = str(pin_memory_device)

    train_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        generator=generator,
        worker_init_fn=worker_fn,
        drop_last=True,
        **_dl_extra,
    )
    val_loader = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        generator=generator,
        worker_init_fn=worker_fn,
        drop_last=False,
        **_dl_extra,
    )
    test_loader = DataLoader(
        te_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        generator=generator,
        worker_init_fn=worker_fn,
        drop_last=False,
        **_dl_extra,
    )
    return train_loader, val_loader, test_loader


def make_loaders_from_h5(
    *,
    train_h5: Path,
    val_h5: Path,
    test_h5: Path,
    batch_size: int,
    seed: int | None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
    pin_memory_device: str = "cuda",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Crea DataLoaders desde H5 (spikes offline)."""
    g = None
    wfn = None
    if seed is not None:
        g = torch.Generator().manual_seed(int(seed))
        wfn = seed_worker

    _HAS_PIN_DEVICE = "pin_memory_device" in inspect.signature(DataLoader).parameters
    _dl_extra: Dict[str, Any] = {}
    if pin_memory and _HAS_PIN_DEVICE and pin_memory_device:
        _dl_extra["pin_memory_device"] = str(pin_memory_device)

    def _mk(path: Path, shuffle: bool) -> DataLoader:
        ds = H5SpikesDataset(path)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=g,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=(prefetch_factor if (num_workers > 0) else None),
            drop_last=False,
            collate_fn=collate_h5,
            worker_init_fn=wfn,
            **_dl_extra,
        )

    return _mk(train_h5, True), _mk(val_h5, False), _mk(test_h5, False)


# ---------------------------------------------------------------------
# Selección de H5 por atributos (enc/T/gain/size/grayscale)
# ---------------------------------------------------------------------
def _pick_h5_by_attrs(base_proc: Path, split: str, *,
                      encoder: str, T: int, gain: float, tfm: ImageTransform) -> Path:
    import h5py
    want_enc  = str(encoder).lower()
    want_T    = int(T)
    want_gain = float(gain) if want_enc == "rate" else 0.0
    want_size = (int(tfm.w), int(tfm.h))
    want_gray = bool(getattr(tfm, "to_gray", True))

    candidates = sorted(base_proc.glob(f"{split}_*.h5"))
    for f in candidates:
        try:
            with h5py.File(f, "r") as h5:
                enc   = str(h5.attrs.get("encoder", "rate")).lower()
                T_h   = int(h5.attrs.get("T", want_T))
                size  = tuple(int(x) for x in h5.attrs.get("size_wh", want_size))
                to_g  = bool(h5.attrs.get("to_gray", 1))
                gain_h= float(h5.attrs.get("gain", 0.0))
                ok = (enc == want_enc) and (T_h == want_T) and (size == want_size) and (to_g == want_gray)
                if enc == "rate":
                    ok = ok and (abs(gain_h - want_gain) < 1e-8)
                if ok:
                    return f
        except Exception:
            pass
    raise FileNotFoundError(
        f"No hay H5 compatible para split={split} en {base_proc}. "
        f"attrs requeridos: enc={want_enc}, T={want_T}, size={want_size}, "
        f"to_gray={want_gray}, gain={want_gain} | Vistos: {[p.name for p in candidates]}"
    )


# ---------------------------------------------------------------------
# Fábrica de loaders según preset (elige H5 offline o CSV + runtime)
# ---------------------------------------------------------------------
def build_make_loader_fn(root: Path, *,
                         use_offline_spikes: bool,
                         encode_runtime: bool):
    """
    Devuelve make_loader_fn(task, batch_size, encoder, T, gain, tfm, seed, **dl_kwargs)
    que construye DataLoaders coherentes con el preset (offline H5 o CSV).
    """
    proc_root = root / "data" / "processed"
    raw_root  = root / "data" / "raw" / "udacity"

    def _abs(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (root / p)

    DL_ALLOWED = {
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
        "shuffle_train", "aug_train",
        "balance_train", "balance_bins", "balance_smooth_eps",
        "pin_memory_device",
    }

    def make_loader_fn(task, batch_size, encoder, T, gain, tfm, seed, **dl_kwargs):
        run = task["name"]

        # Normaliza kwargs
        dl_kwargs = dict(dl_kwargs or {})
        clean_dl = {k: v for k, v in dl_kwargs.items() if (k in DL_ALLOWED and v is not None)}
        if int(clean_dl.get("num_workers", 0)) <= 0:
            clean_dl["persistent_workers"] = False
            clean_dl["prefetch_factor"] = None

        # H5 offline si aplica
        if use_offline_spikes and encoder in {"rate", "latency", "raw"}:
            tr_h5 = _pick_h5_by_attrs(proc_root / run, "train", encoder=encoder, T=T, gain=gain, tfm=tfm)
            va_h5 = _pick_h5_by_attrs(proc_root / run, "val",   encoder=encoder, T=T, gain=gain, tfm=tfm)
            te_h5 = _pick_h5_by_attrs(proc_root / run, "test",  encoder=encoder, T=T, gain=gain, tfm=tfm)
            return make_loaders_from_h5(
                train_h5=tr_h5, val_h5=va_h5, test_h5=te_h5,
                batch_size=batch_size, seed=seed,
                num_workers=clean_dl.get("num_workers", 4),
                pin_memory=clean_dl.get("pin_memory", True),
                persistent_workers=clean_dl.get("persistent_workers", False),
                prefetch_factor=clean_dl.get("prefetch_factor", 2),
                pin_memory_device=clean_dl.get("pin_memory_device", "cuda"),
            )

        # CSV (imágenes) con/ sin runtime encode
        paths   = task["paths"]
        tr_csv  = _abs(paths["train"])
        va_csv  = _abs(paths["val"])
        te_csv  = _abs(paths["test"])
        missing = [p for p in (tr_csv, va_csv, te_csv) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"CSV no encontrado: {missing}. CWD={Path.cwd()} ROOT={root}")

        encoder_for_loader = "image" if (encode_runtime and encoder in {"rate", "latency", "raw"}) else encoder
        return make_loaders_from_csvs(
            base_dir=raw_root / run,
            train_csv=tr_csv, val_csv=va_csv, test_csv=te_csv,
            batch_size=batch_size,
            encoder=encoder_for_loader,
            T=T, gain=gain, tfm=tfm, seed=seed,
            **clean_dl
        )

    return make_loader_fn
