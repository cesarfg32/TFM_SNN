# src/prep/encode_offline.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import cv2

def _imread_center(row, base_dir: Path, to_gray: bool, size_wh: tuple[int,int]) -> np.ndarray:
    """Lee la imagen 'center' del row, reescala y convierte a gris opcionalmente.
    Devuelve HxW (gris) o HxWx3 (RGB) en float32 [0,1].
    """
    # Admite rutas tipo "IMG/xxx.jpg"
    p = (base_dir / str(row["center"])).resolve()
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {p}")
    img = cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)  # size_wh = (W, H)
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # HxW
        img = img.astype(np.float32) / 255.0
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # HxWx3
        img = img.astype(np.float32) / 255.0
    return img

def _encode_rate(img01: np.ndarray, T: int, gain: float, rng: np.random.Generator) -> np.ndarray:
    """Rate coding: Bernoulli(gain*I) por paso temporal.
    img01 puede ser HxW (gris) o HxWxC (RGB). Devuelve:
      - gris ⇒ (T,H,W)  uint8
      - RGB  ⇒ (T,C,H,W) uint8
    """
    if img01.ndim == 2:  # HxW
        H, W = img01.shape
        # mu = I * gain, comparado con uniforme
        u = rng.random((T, H, W), dtype=np.float32)
        spk = (u < (img01[None, :, :] * gain)).astype(np.uint8)
        return spk
    else:  # HxWxC
        H, W, C = img01.shape
        u = rng.random((T, H, W, C), dtype=np.float32)
        spk = (u < (img01[None, :, :, :] * gain)).astype(np.uint8)
        # (T,H,W,C) -> (T,C,H,W)
        return np.transpose(spk, (0, 3, 1, 2))

def _encode_raw(img01: np.ndarray, T: int) -> np.ndarray:
    """RAW: frame estático repetido T veces como spikes binarios (umbral 0.5).
    Conserva canales. Devuelve (T,H,W) o (T,C,H,W).
    """
    if img01.ndim == 2:
        frame = (img01 >= 0.5).astype(np.uint8)
        return np.repeat(frame[None, :, :], T, axis=0)
    else:
        frame = (img01 >= 0.5).astype(np.uint8)  # HxWxC
        frame = np.transpose(frame, (2, 0, 1))   # C,H,W
        return np.repeat(frame[None, :, :, :], T, axis=0)  # T,C,H,W

def _encode_latency(img01: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    """Ejemplo simple de latency: cuanto más brillante el pixel antes dispara.
    Aquí usamos una heurística reproducible. Devuelve float32:
      - gris  ⇒ (T,H,W), con valores {0..T-1} en un canal one-hot temporal (máxima 1 por pixel) o -1 si nada
      - RGB   ⇒ (T,C,H,W) con misma convención.
    Implementación: probamos que el pixel dispare con prob = I, y si dispara le asignamos una latencia
    t ~ Geom(p) truncada en [0, T-1]. Si no dispara => todo -1 (o un -1 en el canal temporal).
    """
    def latency_map(ch01: np.ndarray) -> np.ndarray:
        H, W = ch01.shape
        spikes = np.full((T, H, W), -1.0, dtype=np.float32)  # -1 = sin evento
        p = ch01  # [0,1]
        fire = rng.random((H, W), dtype=np.float32) < p
        # latencia geométrica truncada
        # t = min(geom(p) - 1, T-1); aproximamos geom con -log(u)/log(1-p)
        u = rng.random((H, W), dtype=np.float32)
        denom = np.log(np.clip(1.0 - p, 1e-6, 1.0))
        t = np.minimum((-(np.log(u) / denom)).astype(np.int32), T - 1)
        # set latencia donde fire
        for yy, xx in zip(*np.where(fire)):
            spikes[t[yy, xx], yy, xx] = float(t[yy, xx])
        return spikes

    if img01.ndim == 2:
        return latency_map(img01)
    else:
        H, W, C = img01.shape
        outs = []
        for c in range(C):
            outs.append(latency_map(img01[..., c]))
        # (T,H,W) x C -> (T,C,H,W)
        return np.stack(outs, axis=1)

def encode_csv_to_h5(
    csv_df_or_path: pd.DataFrame | Path | str,
    base_dir: Path,
    out_path: Path,
    *,
    encoder: str = "rate",     # "rate" | "latency" | "raw"
    T: int = 20,
    gain: float = 0.5,
    size_wh: tuple[int, int] = (160, 80),
    to_gray: bool = True,
    seed: int = 42,
    compression: str = "gzip",
) -> None:
    """Lee CSV (columna 'center', 'steering'), codifica frames y guarda H5 con layout oficial:
       - attrs: version=2, encoder, T, gain, size_wh, to_gray, channels
       - datasets: /spikes (N,T,H,W) ó (N,T,C,H,W), /steering (N,), /filenames (N,)
    """
    if not isinstance(csv_df_or_path, pd.DataFrame):
        df = pd.read_csv(csv_df_or_path)
    else:
        df = csv_df_or_path
    assert "center" in df.columns and "steering" in df.columns, "CSV debe tener columnas 'center' y 'steering'"

    rng = np.random.default_rng(seed)
    W, H = int(size_wh[0]), int(size_wh[1])

    # Detecta canales
    first = _imread_center(df.iloc[0], base_dir, to_gray=to_gray, size_wh=(W, H))
    channels = 1 if first.ndim == 2 else first.shape[2]

    N = len(df)
    # Define shapes para /spikes
    if encoder in {"rate", "raw", "latency"}:
        if channels == 1:
            spikes_shape = (N, T, H, W)
            spikes_chunks = (1, 1, H, W)
        else:
            spikes_shape = (N, T, channels, H, W)
            spikes_chunks = (1, 1, channels, H, W)
    else:
        raise ValueError(f"Encoder no soportado: {encoder}")

    # Dtype por encoder
    if encoder in {"rate", "raw"}:
        spikes_dtype = np.uint8
    else:  # latency
        spikes_dtype = np.float32

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        # Atributos
        h5.attrs["version"] = 2
        h5.attrs["encoder"] = encoder
        h5.attrs["T"] = int(T)
        h5.attrs["gain"] = float(gain)
        h5.attrs["size_wh"] = np.asarray([W, H], dtype=np.int32)
        h5.attrs["to_gray"] = int(bool(to_gray))
        h5.attrs["channels"] = int(channels)

        # Datasets
        d_spikes = h5.create_dataset(
            "spikes", shape=spikes_shape, dtype=spikes_dtype,
            chunks=spikes_chunks, compression=compression
        )
        d_y = h5.create_dataset("steering", shape=(N,), dtype=np.float32, compression=compression)
        # Guardamos opcionalmente los nombres de archivo para trazabilidad
        filenames = df["center"].astype(str).to_numpy()
        dt = h5py.string_dtype(encoding="utf-8")
        d_fn = h5.create_dataset("filenames", data=filenames.astype(dt), dtype=dt)

        # Escribir por lotes (aquí fila a fila para simplicidad y memoria acotada)
        for i, row in df.iterrows():
            img01 = _imread_center(row, base_dir, to_gray=to_gray, size_wh=(W, H))
            if encoder == "rate":
                spk = _encode_rate(img01, T=T, gain=gain, rng=rng)
            elif encoder == "raw":
                spk = _encode_raw(img01, T=T)
            elif encoder == "latency":
                spk = _encode_latency(img01, T=T, rng=rng)
            else:
                raise ValueError(encoder)

            # Escribir manteniendo la forma acordada
            if channels == 1:
                # spk: (T,H,W)
                d_spikes[i, :, :, :] = spk
            else:
                # spk: (T,C,H,W)
                d_spikes[i, :, :, :, :] = spk

            d_y[i] = float(row["steering"])
