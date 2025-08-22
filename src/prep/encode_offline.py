# src/prep/encode_offline.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
import pandas as pd

# OpenCV opcional con fallback a PIL
try:
    import cv2
    _HAS_CV2 = True
    try: cv2.setNumThreads(0)
    except: pass
except Exception:
    _HAS_CV2 = False
    from PIL import Image

def _load_image(path: Path, to_gray: bool) -> np.ndarray:
    if _HAS_CV2:
        if to_gray:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None: raise FileNotFoundError(path)
            return img
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with Image.open(path) as im:
        return np.asarray(im.convert("L" if to_gray else "RGB"))

def _resize(arr: np.ndarray, size_wh: tuple[int,int]) -> np.ndarray:
    W, H = size_wh
    if _HAS_CV2:
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
    from PIL import Image
    mode = "L" if arr.ndim == 2 else "RGB"
    return np.asarray(Image.fromarray(arr, mode=mode).resize((W, H), Image.BILINEAR))

def _to_chw01(img: np.ndarray, to_gray: bool) -> np.ndarray:
    if to_gray:
        if img.ndim == 2:
            return (img.astype(np.float32) / 255.0)[None, :, :]
        return (img.mean(-1).astype(np.float32) / 255.0)[None, :, :]
    if img.ndim == 2:
        arr = np.repeat(img[..., None], 3, axis=2).astype(np.float32) / 255.0
    else:
        arr = img.astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))

def encode_rate(x_img: np.ndarray, T: int, gain: float) -> np.ndarray:
    C, H, W = x_img.shape
    p = np.clip(x_img * float(gain), 0.0, 1.0)
    rnd = np.random.random(size=(T, C, H, W))
    return (rnd < p[None, ...]).astype(np.uint8)

def encode_latency(x_img: np.ndarray, T: int) -> np.ndarray:
    C, H, W = x_img.shape
    x1 = np.clip(x_img, 0.0, 1.0)
    t_float = (1.0 - x1) * (T - 1)
    t_idx = np.floor(t_float).astype(np.int64)
    out = np.zeros((T, C, H, W), dtype=np.uint8)
    mask = x1 > 0
    c, h, w = np.where(mask)
    out[t_idx[mask], c, h, w] = 1
    return out

def encode_raw(x_img: np.ndarray, T: int) -> np.ndarray:
    return (x_img[None, ...] > 0).astype(np.uint8).repeat(T, axis=0)

def encode_csv_to_h5(
    *,
    csv_df_or_path,
    base_dir: Path,
    out_path: Path,
    encoder: str,              # "rate" | "latency" | "raw"
    T: int,
    gain: float = 0.5,         # solo "rate"
    size_wh: tuple[int, int] = (160, 80),
    to_gray: bool = True,
    seed: int = 42,
    compression: int = 4,
):
    df = csv_df_or_path if isinstance(csv_df_or_path, pd.DataFrame) else pd.read_csv(csv_df_or_path)
    df = df[[ (base_dir / r["center"]).exists() for _, r in df.iterrows() ]].reset_index(drop=True)

    N = len(df)
    C = 1 if to_gray else 3
    W, H = size_wh
    np.random.seed(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        spikes = h5.create_dataset("spikes", shape=(N, T, C, H, W), dtype="u1",
                                   chunks=(1, T, C, H, W),
                                   compression="gzip", compression_opts=compression)
        steering = h5.create_dataset("steering", shape=(N,), dtype="f4")
        h5.attrs.update({
            "encoder": encoder, "T": T, "gain": (gain if encoder == "rate" else 0.0),
            "size_wh": size_wh, "to_gray": int(to_gray),
        })
        for i, r in df.iterrows():
            img = _load_image((base_dir / r["center"]).resolve(), to_gray)
            img = _resize(img, size_wh)
            chw = _to_chw01(img, to_gray)
            if encoder == "rate":
                spk = encode_rate(chw, T, gain)
            elif encoder == "latency":
                spk = encode_latency(chw, T)
            elif encoder == "raw":
                spk = encode_raw(chw, T)
            else:
                raise ValueError(f"encoder desconocido: {encoder}")
            spikes[i] = spk
            steering[i] = float(r["steering"])
