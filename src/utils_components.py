# src/utils_components.py
# -*- coding: utf-8 -*-
"""
Builders de alto nivel (dataset/modelo/transform) para tools y notebooks.
Devuelve SIEMPRE (make_loader_fn, make_model_fn, tfm) en ese orden.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Callable

from .datasets import ImageTransform, AugmentConfig
from .utils import build_make_loader_fn
from .models import build_model


# Raíz del repo (independiente del CWD al ejecutar)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def build_components_for(cfg: dict) -> Tuple[Callable, Callable, ImageTransform]:
    """
    Construye:
      - tfm: ImageTransform según cfg.model
      - make_loader_fn: DataLoaders coherentes con cfg.data (offline H5 o CSV)
      - make_model_fn: factory del modelo

    Retorno: (make_loader_fn, make_model_fn, tfm)
    """
    # --- Transform ---
    tfm = ImageTransform(
        int(cfg["model"]["img_w"]),
        int(cfg["model"]["img_h"]),
        to_gray=bool(cfg["model"]["to_gray"]),
        crop_top=cfg["model"].get("crop_top", 0) or None,
        crop_bottom=cfg["model"].get("crop_bottom", 0) or None
    )

    # --- Loader factory base ---
    use_offline = bool(cfg["data"].get("use_offline_spikes", False))
    runtime_enc = bool(cfg["data"].get("encode_runtime", False))
    _raw_mk = build_make_loader_fn(
        root=_REPO_ROOT,
        use_offline_spikes=use_offline,
        encode_runtime=runtime_enc,
    )

    # --- Kwargs coherentes con cfg.data ---
    aug_cfg = AugmentConfig(**(cfg["data"].get("aug_train") or {})) if cfg["data"].get("aug_train") else None
    _DL_KW = dict(
        num_workers=int(cfg["data"].get("num_workers") or 0),
        prefetch_factor=int(cfg["data"].get("prefetch_factor") or 2),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", False)),
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

    # >>> Orden estable: (make_loader_fn, make_model_fn, tfm)
    return make_loader_fn, make_model_fn, tfm
