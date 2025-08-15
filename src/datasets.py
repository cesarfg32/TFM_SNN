# src/datasets.py
# -*- coding: utf-8 -*-
"""
Datasets y transformaciones para el TFM (Udacity → SNN).

Puntos clave:
- Carga CSVs del simulador Udacity (center/left/right o un único 'path'/'image').
- Preprocesa imágenes (recorte opcional, redimensionado, escala [0,1], gris opcional).
- Codifica a impulsos on-the-fly:
    - 'rate': Bernoulli por píxel con probabilidad ~ intensidad * gain
    - 'latency': 1 único impulso; cuanto más brillante, más temprano (menor t)
- Devuelve tensores con forma (T, 1, H, W) para que el DataLoader entregue (B, T, 1, H, W).
  En el entrenamiento, permutamos a (T, B, 1, H, W) antes de entrar al modelo.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# -----------------------------
# Transformación de imagen
# -----------------------------
class ImageTransform:
    """Transformación mínima para imágenes de Udacity.

    Args:
        w (int): ancho de destino (por ejemplo, 160).
        h (int): alto de destino (por ejemplo, 80).
        to_gray (bool): si True, convierte a escala de grises (canal único).
        crop_top (Optional[int]): si se indica, recorta 'crop_top' píxeles de la parte superior
                                  antes de redimensionar (útil para eliminar cielo/HUD).
    """

    def __init__(self, w: int, h: int, to_gray: bool = True, crop_top: Optional[int] = None):
        self.w = int(w)
        self.h = int(h)
        self.to_gray = bool(to_gray)
        self.crop_top = crop_top

    def __call__(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Aplica: recorte opcional → conversión (gris o RGB) → resize → normalización → tensor CHW.

        Devuelve:
            torch.Tensor con forma (1, H, W) si to_gray=True; si no, (3, H, W).
        """
        if img_bgr is None:
            raise ValueError("Imagen inválida (cv2.imread devolvió None).")

        # Recorte superior si procede (antes del resize)
        if isinstance(self.crop_top, int) and self.crop_top > 0:
            img_bgr = img_bgr[self.crop_top :, :, :]

        # BGR -> GRAY o BGR -> RGB
        if self.to_gray:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H, W)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # (H, W, 3)

        # Redimensionado a tamaño fijo (w, h)
        interp = cv2.INTER_AREA if (img.shape[0] >= self.h and img.shape[1] >= self.w) else cv2.INTER_LINEAR
        if self.to_gray:
            img = cv2.resize(img, (self.w, self.h), interpolation=interp)  # (h, w)
            img = (img.astype(np.float32) / 255.0)[None, :, :]             # (1, h, w)
        else:
            img = cv2.resize(img, (self.w, self.h), interpolation=interp)  # (h, w, 3)
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)      # (3, h, w)

        return torch.from_numpy(img)  # float32 en [0,1]


# -----------------------------
# Codificadores de impulsos
# -----------------------------
def encode_rate(x_img: torch.Tensor, T: int, gain: float) -> torch.Tensor:
    """Codificación 'rate': para cada píxel, genera T disparos ~ Bernoulli(p), p=clip(gain*I, 0, 1).

    Args:
        x_img: tensor (1, H, W) o (C, H, W) con intensidades en [0,1].
        T: pasos temporales.
        gain: factor de ganancia (escala la intensidad a probabilidad).

    Devuelve:
        Tensor (T, C, H, W) de 0/1 (float32).
    """
    if x_img.dim() == 2:
        x_img = x_img.unsqueeze(0)  # (1,H,W)

    p = (x_img * float(gain)).clamp_(0.0, 1.0)        # (C,H,W)
    # Repite a lo largo de T y samplea Bernoulli
    pT = p.unsqueeze(0).expand(T, *p.shape).contiguous()  # (T,C,H,W)
    # Usamos torch.rand en el mismo device y dtype float32
    rnd = torch.rand_like(pT)
    return (rnd < pT).float()


def encode_latency(x_img: torch.Tensor, T: int) -> torch.Tensor:
    """Codificación 'latency': 1 único impulso por píxel. Más brillo ⇒ spike antes (t más pequeño).

    Estrategia:
      t = floor((1 - I) * (T - 1)), I en [0,1]
      Si I == 0 ⇒ sin disparo (opcional: podríamos no disparar). Aquí disparamos si I > 0.

    Devuelve:
      Tensor (T, C, H, W) con 0/1.
    """
    if x_img.dim() == 2:
        x_img = x_img.unsqueeze(0)  # (1,H,W)

    C, H, W = x_img.shape
    # Calcula el tiempo de disparo por píxel (cuanto mayor I, menor t)
    t_float = (1.0 - x_img.clamp(0.0, 1.0)) * (T - 1)
    t_idx = t_float.floor().to(torch.int64)  # (C,H,W)

    spikes = torch.zeros((T, C, H, W), dtype=torch.float32, device=x_img.device)
    # Máscara de píxeles con I > 0
    mask = x_img > 0.0
    if mask.any():
        # Para cada canal/píxel, coloca un 1 en el tiempo t_idx
        # Usamos índices avanzados: (t, c, h, w)
        t_coords = t_idx[mask]
        c_idx, h_idx, w_idx = torch.where(mask)
        spikes[t_coords, c_idx, h_idx, w_idx] = 1.0
    return spikes


# -----------------------------
# Dataset Udacity
# -----------------------------
@dataclass
class UdacityCSVConfig:
    encoder: str = "rate"   # 'rate' | 'latency' | 'raw'
    T: int = 10
    gain: float = 0.5
    camera: str = "center"  # 'center' | 'left' | 'right'


class UdacityCSV(Dataset):
    """Dataset basado en CSV de Udacity.

    Soporta CSV con columnas:
      - Estándar Udacity: 'center','left','right','steering','throttle','brake','speed'
      - Canonizado por el notebook 01: por ejemplo, 'path','steering' (o 'image','steering').

    Args:
        csv_path: ruta al CSV de split (train/val/test).
        base_dir: carpeta base del recorrido (contiene 'IMG/').
        encoder: 'rate'/'latency'/'raw'.
        T, gain: parámetros del codificador (gain solo aplica a 'rate').
        tfm: transformación de imagen; si None, usa ImageTransform(160,80,True,None).
        camera: si el CSV tiene múltiples rutas (center/left/right), elegir una.
    """

    def __init__(
        self,
        csv_path: Path,
        base_dir: Path,
        encoder: str = "rate",
        T: int = 10,
        gain: float = 0.5,
        tfm: Optional[ImageTransform] = None,
        camera: str = "center",
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.base_dir = Path(base_dir)
        self.cfg = UdacityCSVConfig(encoder=encoder, T=int(T), gain=float(gain), camera=camera)

        # ⚠️ Fallback sano: si no nos pasan tfm, usa valores por defecto razonables (160x80, gris).
        self.tfm = tfm if tfm is not None else ImageTransform(160, 80, True, None)

        assert self.cfg.camera in ["center", "left", "right"], "camera debe ser center/left/right"

        # Carga del CSV una sola vez
        df = pd.read_csv(self.csv_path)

        # Determina la columna de imagen y la de etiqueta (steering)
        self.path_col = self._infer_image_col(df, self.cfg.camera)
        self.label_col = self._infer_label_col(df)

        # Preconstruye lista de (ruta_absoluta, steering) para acelerar __getitem__
        self.samples = []
        for _, row in df.iterrows():
            raw_path = str(row[self.path_col])
            abs_path = self._resolve_path(raw_path)
            y = float(row[self.label_col])
            self.samples.append((abs_path, y))

        if len(self.samples) == 0:
            raise RuntimeError(f"CSV vacío o sin muestras válidas: {self.csv_path}")

    # ---------- utilidades internas ----------

    def _infer_image_col(self, df: pd.DataFrame, camera: str) -> str:
        """Intenta localizar la columna con la ruta de imagen."""
        cols = [c.lower() for c in df.columns]
        # Preferencias típicas de nuestro pipeline
        for candidate in ["path", "image", camera]:
            if candidate in cols:
                # Devuelve el nombre real respetando mayúsculas del CSV
                return df.columns[cols.index(candidate)]
        # Si no encontramos ninguna conocida, intenta la primera que tenga pinta de ruta
        for i, c in enumerate(df.columns):
            if df[c].astype(str).str.contains(r"\.jpg|\.jpeg|\.png", case=False).any():
                return c
        # Último recurso: primera columna
        return df.columns[0]

    def _infer_label_col(self, df: pd.DataFrame) -> str:
        """Localiza la columna de la etiqueta (steering)."""
        cols = [c.lower() for c in df.columns]
        for candidate in ["steering", "angle", "y"]:
            if candidate in cols:
                return df.columns[cols.index(candidate)]
        # Si no existe, error explícito
        raise KeyError(f"No encuentro columna de etiqueta (esperaba 'steering'/'angle'/'y') en {self.csv_path}")

    def _resolve_path(self, raw_path: str) -> str:
        """Convierte una ruta del CSV en una ruta absoluta en disco.

        Casos manejados:
          - Rutas relativas tipo 'IMG/xxx.jpg' → base_dir/'IMG/xxx.jpg'
          - Rutas absolutas de otro PC: intenta recortar desde 'IMG' y reanclar a base_dir/'IMG/...'
          - Backslashes de Windows → se normalizan a '/'
        """
        p = Path(str(raw_path).replace("\\", "/"))

        # Si viene absoluta y contiene 'IMG', recortamos desde 'IMG'
        parts = [q for q in p.parts]
        if p.is_absolute():
            if "IMG" in parts:
                idx = parts.index("IMG")
                p = Path(*parts[idx:])  # 'IMG/xxx.jpg'
            else:
                # Absoluta pero sin 'IMG': último recurso, usar nombre del fichero en base_dir/IMG
                return str(self.base_dir / "IMG" / p.name)

        # Si empieza por 'IMG', ancla a base_dir
        if len(p.parts) >= 1 and p.parts[0] == "IMG":
            return str(self.base_dir / p)

        # Si es relativa pero no empieza por IMG, intenta directamente respecto a base_dir
        return str((self.base_dir / p).resolve())

    # ---------- API Dataset ----------

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Carga con OpenCV en BGR. Si falla, lanza excepción clara."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
        return img

    def _encode(self, x_img: torch.Tensor) -> torch.Tensor:
        """Selecciona el codificador en función de self.cfg.encoder."""
        enc = self.cfg.encoder.lower()
        if enc == "rate":
            return encode_rate(x_img, self.cfg.T, self.cfg.gain)  # (T, C, H, W)
        elif enc == "latency":
            return encode_latency(x_img, self.cfg.T)              # (T, C, H, W)
        elif enc == "raw":
            # Replicar intensidades a lo largo de T (útil para depuración)
            if x_img.dim() == 2:
                x_img = x_img.unsqueeze(0)
            return x_img.unsqueeze(0).expand(self.cfg.T, *x_img.shape).contiguous()
        else:
            raise ValueError(f"Encoder desconocido: {self.cfg.encoder}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Devuelve (X, y):
           - X: (T, 1, H, W)  (spikes 0/1 o intensidades replicadas)
           - y: (1,)          (float32: steering en [-1, +1])
        """
        img_path, y = self.samples[idx]

        # Carga imagen y aplica transformación a tensor (1,H,W) o (3,H,W)
        img_bgr = self._load_image(img_path)
        x_img = self.tfm(img_bgr).float()

        # Para nuestro backbone actual usamos 1 canal; si llegara RGB, reducimos a gris simple
        if x_img.dim() == 3 and x_img.shape[0] == 3:
            # Promedio de canales como conversión rápida; alternativamente cvtColor en el transform
            x_img = x_img.mean(dim=0, keepdim=True)

        # Codificación temporal a impulsos
        X = self._encode(x_img)  # (T, 1, H, W)
        y_t = torch.tensor([y], dtype=torch.float32)

        return X, y_t
