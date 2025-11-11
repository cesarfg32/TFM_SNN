# src/utils.py
# -*- coding: utf-8 -*-
"""Utilidades comunes del proyecto TFM_SNN.

Novedades:
- Opción de balanceo por bins de 'steering' en el split de entrenamiento usando
  WeightedRandomSampler (reduce el sesgo a rectas).
- Opción de pasar configuración de augmentación para el split de entrenamiento.
- Unificación de collate/seed a utilidades globales (picklables) y propagación segura
  de pin_memory_device desde preset (solo si la versión de PyTorch lo soporta).

Retrocompatibilidad:
- Si NO activas 'balance_train' ni pasas 'aug_train', todo se comporta como antes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

import inspect
import random
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import UdacityCSV, ImageTransform, AugmentConfig, H5SpikesDataset
from .loader_utils import collate_h5, seed_worker


# ---------------------------------------------------------------------
# Reproducibilidad global (Python/NumPy/Torch/CUDA)
# ---------------------------------------------------------------------
def set_seeds(seed: int = 42) -> None:
    """Fija semillas para obtener resultados reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Para reproducibilidad estricta (si la necesitas):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Presets de ejecución (fast/std/accurate) desde YAML
# ---------------------------------------------------------------------
def load_preset(path_yaml: Path, name: str) -> dict:
    with open(path_yaml, "r", encoding="utf-8") as f:
        presets = yaml.safe_load(f)
    assert name in presets, f"Preset no encontrado: {name} en {path_yaml}"
    cfg = presets[name]
    # Validación mínima
    for k in ("model", "data", "optim", "continual", "naming"):
        assert k in cfg, f"Preset '{name}' incompleto: falta '{k}'"
    return cfg


# ---------------------------------------------------------------------
# Sampler balanceado por bins para entrenamiento
# ---------------------------------------------------------------------
def _make_balanced_sampler(labels: List[float], bins: int = 21, smooth_eps: float = 1e-3) -> WeightedRandomSampler:
    """
    Crea un WeightedRandomSampler que muestrea de forma aproximadamente uniforme
    por bins de 'steering'.
    """
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
# DataLoaders con opción de siembra, augmentación y balanceo (train)
# ---------------------------------------------------------------------
def make_loaders_from_csvs(
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
    num_workers: int = 8,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 4,
    # --- Novedades ---
    aug_train: Optional[AugmentConfig] = None,
    balance_train: bool = False,
    balance_bins: int = 50,
    balance_smooth_eps: float = 1e-3,
    # pin memory device (se propaga si la versión de DataLoader lo soporta)
    pin_memory_device: str = "cuda",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train/val/test del simulador Udacity con codificación temporal on-the-fly.
    """
    # Datasets
    tr_ds = UdacityCSV(train_csv, base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm, aug=aug_train)
    va_ds = UdacityCSV(val_csv,   base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm, aug=None)
    te_ds = UdacityCSV(test_csv,  base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm, aug=None)

    # Siembra opcional: controla orden del sampler/shuffle y estados de los workers
    generator = None
    worker_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_fn = seed_worker  # <- global

    # Sampler balanceado (solo train) — si se activa, desactiva shuffle_train
    train_sampler = None
    effective_shuffle = bool(shuffle_train)
    if balance_train:
        train_sampler = _make_balanced_sampler(
            tr_ds.labels, bins=balance_bins, smooth_eps=balance_smooth_eps
        )
        effective_shuffle = False

    # Normaliza flags del DataLoader según num_workers
    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None
    else:
        if prefetch_factor is None:
            prefetch_factor = 2

    # Soporte condicional de pin_memory_device (solo si la firma de DataLoader lo acepta)
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
        **_dl_extra,
    )
    return train_loader, val_loader, test_loader


def make_loaders_from_h5(
    train_h5: Path,
    val_h5: Path,
    test_h5: Path,
    *,
    batch_size: int,
    seed: int | None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = 2,
    pin_memory_device: str = "cuda",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Crea DataLoaders desde H5 (spikes offline) con orientación (T,B,C,H,W) en el batch."""
    # Siembra opcional para reproducibilidad
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
            collate_fn=collate_h5,  # global y picklable
            worker_init_fn=wfn,
            **_dl_extra,
        )

    return _mk(train_h5, True), _mk(val_h5, False), _mk(test_h5, False)


def _pick_h5_by_attrs(base_proc: Path, split: str, *, encoder: str, T: int, gain: float, tfm: ImageTransform) -> Path:
    """Selecciona el .h5 que casa con (encoder, T, gain, tamaño y gris/color del transform)."""
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


def build_make_loader_fn(root: Path, *, use_offline_spikes: bool, encode_runtime: bool):
    """Devuelve make_loader_fn(...) que elige entre H5 offline o CSV + runtime encode."""
    proc_root = root / "data" / "processed"
    raw_root  = root / "data" / "raw" / "udacity"

    def _abs(p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (root / p)

    # claves permitidas para DataLoaders
    DL_ALLOWED = {
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
        "shuffle_train", "aug_train",
        "balance_train", "balance_bins", "balance_smooth_eps",
        "pin_memory_device",  # ← ahora permitido
    }

    def make_loader_fn(task, batch_size, encoder, T, gain, tfm, seed, **dl_kwargs):
        run = task["name"]
        base_raw  = raw_root / run

        # Normaliza kwargs del DataLoader (única fuente de verdad)
        dl_kwargs = dict(dl_kwargs or {})
        clean_dl = {k: v for k, v in dl_kwargs.items() if (k in DL_ALLOWED and v is not None)}

        if int(clean_dl.get("num_workers", 0)) <= 0:
            clean_dl["persistent_workers"] = False
            clean_dl["prefetch_factor"] = None

        # --- H5 offline ---
        if use_offline_spikes and encoder in {"rate", "latency", "raw"}:
            tr_h5 = _pick_h5_by_attrs(proc_root / run, "train", encoder=encoder, T=T, gain=gain, tfm=tfm)
            va_h5 = _pick_h5_by_attrs(proc_root / run, "val",   encoder=encoder, T=T, gain=gain, tfm=tfm)
            te_h5 = _pick_h5_by_attrs(proc_root / run, "test",  encoder=encoder, T=T, gain=gain, tfm=tfm)
            return make_loaders_from_h5(
                train_h5=tr_h5, val_h5=va_h5, test_h5=te_h5,
                batch_size=batch_size, seed=seed,
                num_workers=clean_dl.get("num_workers", 4),
                pin_memory=clean_dl.get("pin_memory", True),
                persistent_workers=clean_dl.get("persistent_workers", True),
                prefetch_factor=clean_dl.get("prefetch_factor", 2),
                pin_memory_device=clean_dl.get("pin_memory_device", "cuda"),
            )

        # --- CSV (imágenes) con/ sin runtime encode ---
        paths    = task["paths"]
        tr_csv = _abs(paths["train"])
        va_csv = _abs(paths["val"])
        te_csv = _abs(paths["test"])
        missing = [p for p in (tr_csv, va_csv, te_csv) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"CSV no encontrado: {missing}. CWD={Path.cwd()} ROOT={root}")

        encoder_for_loader = "image" if (encode_runtime and encoder in {"rate","latency","raw"}) else encoder
        return make_loaders_from_csvs(
            base_dir=base_raw,
            train_csv=tr_csv, val_csv=va_csv, test_csv=te_csv,
            batch_size=batch_size,
            encoder=encoder_for_loader,
            T=T, gain=gain, tfm=tfm, seed=seed,
            **clean_dl
        )

    return make_loader_fn


# ---------------------------------------------------------------------
# Helpers comunes para notebooks/tools: task_list y factories por preset
# ---------------------------------------------------------------------
def build_task_list_for(cfg: dict, root: Path):
    """Devuelve (task_list, tasks_file_path) respetando 'prep.use_balanced_tasks'."""
    proc = root / "data" / "processed"
    use_bal = bool(cfg.get("prep", {}).get("use_balanced_tasks", False))
    tb_name = (cfg.get("prep", {}).get("tasks_balanced_file_name") or "tasks_balanced.json")
    t_name  = (cfg.get("prep", {}).get("tasks_file_name") or "tasks.json")

    tasks_file = (proc / tb_name) if (use_bal and (proc / tb_name).exists()) else (proc / t_name)
    if not tasks_file.exists():
        raise FileNotFoundError(
            f"No existe {tasks_file.name} en {proc}. Genera los splits con tu pipeline "
            f"(p.ej. tools/prep_offline.py) antes de entrenar."
        )
    tasks_json = json.loads(tasks_file.read_text(encoding="utf-8"))
    task_list = [{"name": n, "paths": tasks_json["splits"][n]} for n in tasks_json["tasks_order"]]
    return task_list, tasks_file


def build_components_for(cfg: dict, root: Path):
    """
    Construye:
      - tfm: ImageTransform coherente con el preset
      - make_loader_fn: elige H5 offline o CSV + runtime encode
      - make_model_fn: factory de modelo
    """
    # import diferido para evitar ciclos con src.models
    from src.models import build_model

    # --- Transform ---
    tfm = ImageTransform(
        int(cfg["model"]["img_w"]),
        int(cfg["model"]["img_h"]),
        to_gray=bool(cfg["model"]["to_gray"]),
        crop_top=None
    )

    # --- Loader factory base ---
    use_offline = bool(cfg["data"].get("use_offline_spikes", False))
    runtime_enc = bool(cfg["data"].get("encode_runtime", False))
    _raw_mk = build_make_loader_fn(root=root, use_offline_spikes=use_offline, encode_runtime=runtime_enc)

    # --- Kwargs coherentes con cfg ---
    aug_cfg = AugmentConfig(**(cfg["data"].get("aug_train") or {})) if cfg["data"].get("aug_train") else None
    _DL_KW = dict(
        num_workers=int(cfg["data"].get("num_workers") or 0),
        prefetch_factor=int(cfg["data"].get("prefetch_factor") or 2),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)),
        pin_memory_device=str(cfg["data"].get("pin_memory_device", "cuda")),
        aug_train=aug_cfg,
        balance_train=bool(cfg["data"].get("balance_online", False)),
        balance_bins=int(cfg["data"].get("balance_bins") or 50),
        balance_smooth_eps=float(cfg["data"].get("balance_smooth_eps") or 1e-3),
    )

    def make_loader_fn(task, batch_size, encoder, T, gain, tfm, seed, **dl_kwargs):
        return _raw_mk(
            task=task, batch_size=batch_size, encoder=encoder, T=T, gain=gain,
            tfm=tfm, seed=seed, **{**_DL_KW, **(dl_kwargs or {})}
        )

    def make_model_fn(tfm_):
        return build_model(cfg["model"]["name"], tfm_, beta=0.9, threshold=0.5)

    return tfm, make_loader_fn, make_model_fn
