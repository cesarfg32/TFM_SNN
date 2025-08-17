# -*- coding: utf-8 -*-
"""
Utilidades comunes del proyecto TFM_SNN.

Novedades:
- Opción de balanceo por bins de 'steering' en el split de entrenamiento usando
  WeightedRandomSampler (reduce el sesgo a rectas).
- Opción de pasar configuración de augmentación para el split de entrenamiento.

Retrocompatibilidad:
- Si NO activas 'balance_train' ni pasas 'aug_train', todo se comporta como antes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import UdacityCSV, ImageTransform, AugmentConfig


# ---------------------------------------------------------------------
# Reproducibilidad global (Python/NumPy/Torch/CUDA)
# ---------------------------------------------------------------------
def set_seeds(seed: int = 42) -> None:
    """Fija semillas para obtener resultados reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Para que cuDNN no haga auto-tuning que cambie el orden de operaciones
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Presets de ejecución (fast/std/accurate) desde YAML
# ---------------------------------------------------------------------
def load_preset(path_yaml: Path, name: str):
    """Carga un preset por nombre desde un YAML (p. ej., configs/presets.yaml)."""
    with open(path_yaml, "r", encoding="utf-8") as f:
        presets = yaml.safe_load(f)
    assert name in presets, f"Preset no encontrado: {name} en {path_yaml}"
    return presets[name]


# ---------------------------------------------------------------------
# Inicialización determinista por worker
# ---------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """
    Inicializa de forma determinista cada worker del DataLoader.

    Recomendado por PyTorch:
    - Derivar semilla del worker a partir de torch.initial_seed() (ajustada a 32 bits).
    - Con esa semilla inicializar NumPy y random.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# Sampler balanceado por bins para entrenamiento
# ---------------------------------------------------------------------
def _make_balanced_sampler(labels: list[float], bins: int = 21, smooth_eps: float = 1e-3) -> WeightedRandomSampler:
    """
    Crea un WeightedRandomSampler que muestrea de forma aproximadamente uniforme por bins de 'steering'.

    - 'bins' define la discretización del continuo [-1,1] (o del rango observado).
    - Las muestras de bins con menos frecuencia reciben más peso (1 / count_bin).
    - Se usa 'replacement=True' y 'num_samples=len(labels)' para mantener el mismo número de pasos por época.
    """
    y = np.asarray(labels, dtype=np.float32)
    lo = min(-1.0, float(y.min()))
    hi = max( 1.0, float(y.max()))
    edges = np.linspace(lo, hi, int(bins))
    # Bin de cada muestra (valores en [0, bins-2] pues edges tiene 'bins' puntos y 'bins-1' intervalos)
    bin_ids = np.clip(np.digitize(y, edges) - 1, 0, len(edges) - 2)

    # Conteos por bin y pesos inversos
    counts = np.bincount(bin_ids, minlength=len(edges) - 1).astype(np.float32)
    counts = counts + float(smooth_eps)  # suavizado para evitar división por cero
    inv = 1.0 / counts
    # Peso por muestra = peso del bin correspondiente
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
    prefetch_factor: int = 4,
    # --- Novedades ---
    aug_train: Optional[AugmentConfig] = None,
    balance_train: bool = False,
    balance_bins: int = 21,
    balance_smooth_eps: float = 1e-3,
):
    """
    Crea DataLoaders para train/val/test del simulador Udacity con codificación temporal on-the-fly.

    Rendimiento:
    - Con `pin_memory=True`, usa `tensor.to(device, non_blocking=True)` para solapar transferencias CPU→GPU.
    - `persistent_workers=True` evita relanzar los workers en cada época (menos overhead).
    - `prefetch_factor` controla cuántos batches prepara cada worker por adelantado.

    Compatibilidad:
    - Si NO pasas `seed`, el orden de muestreo será no determinista (comportamiento anterior).
    - Si pasas `seed`, se fija el orden del sampler y la semilla de cada worker (resultados reproducibles).
    - Si NO pasas `aug_train` ni `balance_train`, el comportamiento es idéntico al actual.

    Args:
        base_dir (Path): Carpeta base del recorrido (contiene 'IMG/').
        train_csv / val_csv / test_csv (Path): Rutas a los CSV de cada split.
        batch_size (int): Tamaño de batch.
        encoder (str): 'rate' | 'latency' | 'raw'.
        T (int): Ventana temporal (nº de pasos).
        gain (float): Ganancia (aplica a 'rate').
        tfm (ImageTransform): Transformación de imagen (p. ej., ImageTransform(160,80,True,None)).
        seed (int | None): Si se indica, hace reproducible el orden del DataLoader.
        num_workers (int): Nº de workers por DataLoader (por defecto 8).
        pin_memory (bool): Activa pinned memory en CUDA (por defecto True).
        shuffle_train (bool): Si baraja el split de entrenamiento (por defecto True).
        persistent_workers (bool): Mantiene vivos los workers entre épocas (por defecto True).
        prefetch_factor (int): Nº de batches prefetech por worker (por defecto 4).

        # Novedades:
        aug_train (AugmentConfig | None): augmentación SOLO en train.
        balance_train (bool): activa sampler balanceado por bins de 'steering' (train).
        balance_bins (int): número de bins para el balanceo.
        balance_smooth_eps (float): suavizado para los conteos por bin.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    # --- Guardia anti doble balanceo ---
    if balance_train and Path(train_csv).name == "train_balanced.csv":
        print("[WARN] CSV ya balanceado offline detectado; desactivo balance_train para evitar doble balanceo.")
        balance_train = False

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
        worker_fn = _seed_worker

    # Sampler balanceado (solo train) — si se activa, desactiva shuffle_train
    train_sampler = None
    effective_shuffle = bool(shuffle_train)
    if balance_train:
        train_sampler = _make_balanced_sampler(tr_ds.labels, bins=balance_bins, smooth_eps=balance_smooth_eps)
        effective_shuffle = False  # DataLoader no permite 'shuffle' junto con 'sampler'

    # Normaliza flags del DataLoader según num_workers
    if num_workers <= 0:
        persistent_workers = False
        prefetch_factor = None
    else:
        if prefetch_factor is None:
            prefetch_factor = 2  # valor mínimo válido cuando hay workers

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
    )
    return train_loader, val_loader, test_loader
