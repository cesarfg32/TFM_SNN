# src/datasets.py
# -*- coding: utf-8 -*-
"""Datasets y transformaciones para el TFM (Udacity → SNN).

Novedades:
- Ruta rápida con OpenCV para decodificar/resize (más veloz).
- Fallback a PIL si no hay OpenCV o falla la lectura.
- Augmentación opcional (flip con corrección, brillo, gamma, ruido).
- **Respeto estricto de `to_gray`**: si el transform va en gris ⇒ 1 canal; si va en color ⇒ 3 canales.

Puntos clave:
- Lee CSVs (center/left/right o path/image) del simulador Udacity.
- Preprocesa imágenes (crop top opcional, resize, normalización [0,1], gris opcional).
- Codifica a impulsos on-the-fly:
  - 'rate': Bernoulli por píxel ~ intensidad*gain
  - 'latency': 1 spike; más brillo ⇒ antes (t menor)
- Devuelve tensores (T, C, H, W). C=1 si gris, C=3 si color.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---- OpenCV rápido si existe; si no, fallback a PIL ----
try:
    import cv2
    _HAS_CV2 = True
    try:
        cv2.setNumThreads(0)  # evita sobre-subscription
    except Exception:
        pass
except Exception:
    cv2 = None
    _HAS_CV2 = False

from PIL import Image

# -----------------------------
# Transformación de imagen
# -----------------------------
class ImageTransform:
    """Transformación mínima para imágenes de Udacity.
    Args:
      w (int): ancho destino (p. ej. 160).
      h (int): alto destino (p. ej. 80).
      to_gray (bool): si True, salida en escala de grises (1 canal); si False, RGB (3 canales).
      crop_top (Optional[int]): recorta 'crop_top' px por arriba antes del resize.
    """
    def __init__(self, w: int, h: int, to_gray: bool = True, crop_top: Optional[int] = None):
        self.w = int(w)
        self.h = int(h)
        self.to_gray = bool(to_gray)
        self.crop_top = crop_top

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        if img_bgr is None:
            raise ValueError("Imagen inválida (None).")
        # Crop superior
        if isinstance(self.crop_top, int) and self.crop_top > 0:
            img_bgr = img_bgr[self.crop_top:, ...]

        if _HAS_CV2:
            if self.to_gray:
                if img_bgr.ndim == 3:
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H,W)
                else:
                    img = img_bgr
                img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)  # (h,w)
                img = (img.astype(np.float32) / 255.0)[None, :, :]  # (1,h,w)
            else:
                if img_bgr.ndim == 3:
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # (H,W,3)
                else:
                    img = np.repeat(img_bgr[..., None], 3, axis=2)  # (H,W,3)
                img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)  # (h,w,3)
                img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3,h,w)
            return torch.from_numpy(img)

        # Fallback PIL
        if self.to_gray:
            if img_bgr.ndim == 3:
                b, g, r = img_bgr[..., 0], img_bgr[..., 1], img_bgr[..., 2]
                img = (0.114*b + 0.587*g + 0.299*r).astype(np.uint8)
            else:
                img = img_bgr.astype(np.uint8)
            pil = Image.fromarray(img, mode="L")
            pil = pil.resize((self.w, self.h), resample=Image.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32) / 255.0  # (h,w)
            arr = arr[None, :, :]  # (1,h,w)
        else:
            if img_bgr.ndim == 2:
                img = np.repeat(img_bgr[..., None], 3, axis=2)
            else:
                img = img_bgr[..., ::-1]  # BGR -> RGB
            pil = Image.fromarray(img, mode="RGB")
            pil = pil.resize((self.w, self.h), resample=Image.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32) / 255.0  # (h,w,3)
            arr = np.transpose(arr, (2, 0, 1))  # (3,h,w)
        return torch.from_numpy(arr)

# -----------------------------
# Augmentación de imagen/label
# -----------------------------
@dataclass
class AugmentConfig:
    """Config de augmentación para UdacityCSV (solo en train)."""
    prob_hflip: float = 0.0
    brightness: Optional[Tuple[float, float]] = None
    gamma: Optional[Tuple[float, float]] = None
    noise_std: float = 0.0

# -----------------------------
# Codificadores (CPU, para usar en Dataset si hace falta)
# -----------------------------
def encode_rate(x_img: torch.Tensor, T: int, gain: float) -> torch.Tensor:
    if x_img.dim() == 2:
        x_img = x_img.unsqueeze(0)
    p = (x_img * float(gain)).clamp_(0.0, 1.0)
    pT = p.unsqueeze(0).expand(T, *p.shape).contiguous()
    rnd = torch.rand_like(pT)
    return (rnd < pT).float()

def encode_latency(x_img: torch.Tensor, T: int) -> torch.Tensor:
    if x_img.dim() == 2:
        x_img = x_img.unsqueeze(0)
    C, H, W = x_img.shape
    t_float = (1.0 - x_img.clamp(0.0, 1.0)) * (T - 1)
    t_idx = t_float.floor().to(torch.int64)
    spikes = torch.zeros((T, C, H, W), dtype=torch.float32, device=x_img.device)
    mask = x_img > 0.0
    if mask.any():
        t_coords = t_idx[mask]
        c_idx, h_idx, w_idx = torch.where(mask)
        spikes[t_coords, c_idx, h_idx, w_idx] = 1.0
    return spikes

# -----------------------------
# Dataset Udacity
# -----------------------------
@dataclass
class UdacityCSVConfig:
    encoder: str = "rate"  # 'rate' | 'latency' | 'raw' | 'image'
    T: int = 10
    gain: float = 0.5
    camera: str = "center"  # 'center' | 'left' | 'right'

class UdacityCSV(Dataset):
    """Dataset basado en CSV de Udacity."""
    def __init__(
        self,
        csv_path: Path,
        base_dir: Path,
        encoder: str = "rate",
        T: int = 10,
        gain: float = 0.5,
        tfm: Optional[ImageTransform] = None,
        camera: str = "center",
        aug: Optional[AugmentConfig] = None,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.base_dir = Path(base_dir)
        self.cfg = UdacityCSVConfig(encoder=encoder, T=int(T), gain=float(gain), camera=camera)
        self.aug = aug
        self.tfm = tfm if tfm is not None else ImageTransform(160, 80, True, None)
        assert self.cfg.camera in ["center", "left", "right"], "camera debe ser center/left/right"

        df = pd.read_csv(self.csv_path)
        self.path_col = self._infer_image_col(df, self.cfg.camera)
        self.label_col = self._infer_label_col(df)

        self.samples = []
        for _, row in df.iterrows():
            raw_path = str(row[self.path_col])
            abs_path = self._resolve_path(raw_path)
            y = float(row[self.label_col])
            self.samples.append((abs_path, y))
        if len(self.samples) == 0:
            raise RuntimeError(f"CSV vacío o sin muestras válidas: {self.csv_path}")
        self.labels = [y for _, y in self.samples]

    # ---------- utils internos ----------
    def _infer_image_col(self, df: pd.DataFrame, camera: str) -> str:
        cols = [c.lower() for c in df.columns]
        for candidate in ["path", "image", camera]:
            if candidate in cols:
                return df.columns[cols.index(candidate)]
        for c in df.columns:
            if df[c].astype(str).str.contains(r"\.(jpg|jpeg|png)$", case=False, regex=True).any():
                return c
        return df.columns[0]

    def _infer_label_col(self, df: pd.DataFrame) -> str:
        cols = [c.lower() for c in df.columns]
        for candidate in ["steering", "angle", "y"]:
            if candidate in cols:
                return df.columns[cols.index(candidate)]
        raise KeyError(f"No encuentro columna de etiqueta (steering/angle/y) en {self.csv_path}")

    def _resolve_path(self, raw_path: str) -> str:
        p = Path(str(raw_path).replace("\\", "/"))
        if p.is_absolute():
            parts = [q for q in p.parts]
            if "IMG" in parts:
                idx = parts.index("IMG")
                p = Path(*parts[idx:])
            else:
                return str(self.base_dir / "IMG" / p.name)
        if len(p.parts) >= 1 and p.parts[0] == "IMG":
            return str(self.base_dir / p)
        return str((self.base_dir / p).resolve())

    # ---------- carga de imagen ----------
    def _load_image(self, path: str, as_gray: bool | None = None) -> np.ndarray:
        if as_gray is None:
            as_gray = False
        if _HAS_CV2:
            flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
            img = cv2.imread(path, flag)
            if img is not None:
                return img
        with Image.open(path) as im:
            if as_gray:
                im = im.convert("L")
                arr = np.asarray(im)
                return arr
            else:
                im = im.convert("RGB")
                rgb = np.asarray(im)
                bgr = rgb[..., ::-1].copy()
                return bgr

    def _encode(self, x_img: torch.Tensor) -> torch.Tensor:
        enc = self.cfg.encoder.lower()
        if enc == "image":
            # Devolver 4D (B,C,H,W) a nivel de batch; aquí solo devolvemos (C,H,W)
            return x_img  # sin temporal
        elif enc == "rate":
            return encode_rate(x_img, self.cfg.T, self.cfg.gain)
        elif enc == "latency":
            return encode_latency(x_img, self.cfg.T)
        elif enc == "raw":
            if x_img.dim() == 2:
                x_img = x_img.unsqueeze(0)
            return x_img.unsqueeze(0).expand(self.cfg.T, *x_img.shape).contiguous()
        else:
            raise ValueError(f"Encoder desconocido: {self.cfg.encoder}")

    # ---------- augment ----------
    def _apply_augmentation(self, x_img: torch.Tensor, y: float) -> tuple[torch.Tensor, float]:
        if self.aug is None:
            return x_img, y
        if self.aug.brightness is not None:
            lo, hi = self.aug.brightness
            if hi > 0 and hi != 1.0:
                factor = random.uniform(float(lo), float(hi))
                x_img = (x_img * factor).clamp_(0.0, 1.0)
        if self.aug.gamma is not None:
            g_lo, g_hi = self.aug.gamma
            if g_hi > 0:
                gamma = max(1e-6, random.uniform(float(g_lo), float(g_hi)))
                x_img = x_img.pow(gamma).clamp_(0.0, 1.0)
        if self.aug.noise_std and self.aug.noise_std > 0:
            noise = torch.randn_like(x_img) * float(self.aug.noise_std)
            x_img = (x_img + noise).clamp_(0.0, 1.0)
        if self.aug.prob_hflip > 0 and random.random() < self.aug.prob_hflip:
            x_img = torch.flip(x_img, dims=[2])  # invierte ancho
            y = -float(y)
        return x_img, y

    # ---------- API Dataset ----------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, y = self.samples[idx]
        img_arr = self._load_image(img_path, as_gray=self.tfm.to_gray)
        x_img = self.tfm(img_arr).float()

        # Respeta to_gray
        if self.tfm.to_gray:
            if x_img.dim() == 2:
                x_img = x_img.unsqueeze(0)
            elif x_img.dim() == 3 and x_img.shape[0] == 3:
                x_img = x_img.mean(dim=0, keepdim=True)
        else:
            if x_img.dim() == 2:
                x_img = x_img.unsqueeze(0).expand(3, *x_img.shape[1:])

        # Augment (solo para 'image' o 'raw' antes de codificar)
        x_img, y = self._apply_augmentation(x_img, y)

        X = self._encode(x_img)  # (T,C,H,W) o (C,H,W) si encoder='image'
        y_t = torch.tensor([y], dtype=torch.float32)
        return X, y_t

# Dataset para H5 con spikes offline
import h5py

class H5SpikesDataset(Dataset):
    """
    Lee archivos H5 con estructura:
      - Atributos: T, encoder, (gain), size_wh, to_gray
      - Datasets:  'spikes'  (N, T, H, W)   o (N, T, C, H, W)
                   'steering' (N,)
    Devuelve: (x, y) con x -> (T, C, H, W) float32, y -> tensor([float32])
    """
    def __init__(self, h5_path: Path | str):
        super().__init__()
        self.h5_path = str(h5_path)
        self._h5 = None
        self._spikes = None
        self._steer = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._spikes = self._h5["spikes"]
            self._steer  = self._h5["steering"]

    def __len__(self):
        self._ensure_open()
        return self._spikes.shape[0]

    def __getitem__(self, i: int):
        self._ensure_open()
        x = self._spikes[i]  # (T,H,W) o (T,C,H,W)
        if x.ndim == 3:      # (T,H,W) -> (T,1,H,W)
            x = x[None, ...]           # (1,T,H,W)
            x = x.transpose(1, 0, 2, 3) # (T,1,H,W)
        x_t = torch.from_numpy(x)
        if x_t.dtype == torch.uint8:
            x_t = x_t.float()
        y = float(self._steer[i])
        y_t = torch.tensor([y], dtype=torch.float32)
        return x_t, y_t

    def close(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
