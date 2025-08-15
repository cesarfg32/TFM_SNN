# -*- coding: utf-8 -*-
"""
Utilidades comunes del proyecto TFM_SNN.

Recomendaciones de rendimiento (GPU RTX 4080 / CUDA):
- TF32 en FP32 (solo una vez al arrancar, p. ej., en src/training.py):
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

- DataLoader (alimentar bien a la GPU):
    num_workers=8   (ajusta según tu CPU)
    pin_memory=True
    persistent_workers=True
    prefetch_factor=4

- Transferencias CPU→GPU:
    tensor = tensor.to(device, non_blocking=True)  # requiere pin_memory=True

- Reproducibilidad vs. velocidad:
    * Reproducible:
        set_seeds(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    * Rápido (no determinista):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import UdacityCSV, ImageTransform


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# DataLoaders con opción de siembra (orden reproducible entre corridas)
# ---------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """
    Inicializa de forma determinista cada worker del DataLoader.

    Por qué:
        Cada worker es un proceso distinto. Si no se inicializa,
        NumPy y `random` podrían compartir estado entre workers y
        generar órdenes distintos entre corridas, rompiendo la
        reproducibilidad incluso con `set_seeds`.

    Implementación (recomendada por PyTorch):
        - Derivamos la semilla específica del worker a partir de
          `torch.initial_seed()` y la acotamos a 32 bits.
        - Con esa semilla inicializamos NumPy y `random`.

    Uso:
        - Pasa `worker_init_fn=_seed_worker` al crear el DataLoader.
        - Para fijar además el orden del *sampler* (shuffle), crea
          un `generator = torch.Generator(); generator.manual_seed(seed)`
          y pásalo al DataLoader junto con esta función.

    Nota:
        Esto complementa a `generator.manual_seed(seed)` (controla el
        muestreador/shuffle). Aquí nos aseguramos también de que las
        libs del worker (NumPy/`random`) queden en el mismo estado.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    # Datasets (se crean siempre igual)
    tr_ds = UdacityCSV(train_csv, base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm)
    va_ds = UdacityCSV(val_csv,   base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm)
    te_ds = UdacityCSV(test_csv,  base_dir, encoder=encoder, T=T, gain=gain, tfm=tfm)

    # Siembra opcional: garantiza mismo orden entre corridas y entre métodos (naive vs ewc)
    generator = None
    worker_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))   # controla el orden del shuffle y los samplers
        worker_fn = _seed_worker           # fija NumPy/random en cada worker

    train_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=bool(shuffle_train),
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
