# src/prep/encode_offline.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import h5py
import cv2

def _upsert_prep_manifest(manifest_path: Path, key: str, entry: dict) -> None:
    """Crea/actualiza prep_manifest.json sin borrar otras entradas (atomic write)."""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except Exception:
        # Si está corrupto, re-inicializamos para no bloquear el pipeline
        data = {}
    data[key] = entry
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(manifest_path)


def _imread_center_mixed(
    row,
    base_raw: Path,
    base_proc: Path,
    to_gray: bool,
    size_wh: tuple[int, int],
    crop_top: int = 0,
    crop_bottom: int = 0,
) -> np.ndarray:
    """Lee la imagen indicada en row['center'] resolviendo:
       - 'aug/train/...'  -> bajo processed/<run>
       - 'IMG/...'/otras  -> bajo raw/<run>
       Aplica crop superior/inferior opcional y resize a (W,H).
       Devuelve HxW (gris) o HxWx3 (RGB) en float32 [0,1].
    """
    rel = str(row["center"]).replace("\\", "/").lstrip("/")
    # Heurística: si empieza por 'aug/' usamos processed; si no, raw
    root = base_proc if rel.lower().startswith("aug/") else base_raw
    p = (root / rel).resolve()

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {p}")

    # Crop top / bottom antes del resize
    top = max(0, int(crop_top or 0))
    bot = max(0, int(crop_bottom or 0))
    if top > 0 or bot > 0:
        h = img.shape[0]
        start = min(top, h)
        end = h - bot if bot > 0 else h
        end = max(start + 1, end)
        img = img[start:end, :, :]

    W, H = size_wh  # nota: en este módulo usamos (W,H)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

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
    """
    Latency coding robusta para SNN.

    Idea original (conservada y comentada):
      - Cuanto más brillante el píxel, antes dispara (latencia menor).
      - Usamos una heurística reproducible: con prob. p = I (intensidad en [0,1])
        el píxel dispara; si dispara, su latencia es t = floor((1 - I)*(T-1)).
      - Si no dispara, no hay evento (todo el vector temporal a 0.0).

    Diferencias respecto a la versión anterior (arreglos):
      - Clamp de intensidades y de índices: t ∈ [0, T-1] ⇒ sin índices negativos.
      - Se escribe 1.0 en el instante t (one-hot temporal), no el valor de t.
      - Soporta entradas (H,W), (C,H,W) o (H,W,C) y devuelve (T,H,W) o (T,C,H,W).

    Parámetros:
      img01 : np.ndarray float32 normalizada a [0,1]
      T     : tamaño temporal
      rng   : np.random.Generator para reproducibilidad

    Devuelve:
      np.ndarray con spikes 0/1:
        - (T,H,W)  si entrada era (H,W)
        - (T,C,H,W) si entrada tenía canales
    """
    # --- Normalización de forma y dominio ---
    arr = np.asarray(img01, dtype=np.float32)
    np.clip(arr, 0.0, 1.0, out=arr)

    if arr.ndim == 2:                 # (H,W)
        C, H, W = 1, arr.shape[0], arr.shape[1]
        arr_c = arr[None, ...]        # -> (1,H,W)
    elif arr.ndim == 3:
        # Acepta (C,H,W) o (H,W,C)
        if arr.shape[0] in (1, 3):    # asumimos (C,H,W)
            C, H, W = arr.shape
            arr_c = arr
        else:                         # (H,W,C) -> (C,H,W)
            H, W, C = arr.shape
            arr_c = np.transpose(arr, (2, 0, 1)).copy()
    else:
        raise ValueError(f"Forma inesperada: {arr.shape}")

    # --- Reserva salida (T,C,H,W) ---
    spikes = np.zeros((T, C, H, W), dtype=np.float32)

    # --- Máscara de disparo (prob. p = I) ---
    u_fire = rng.random((C, H, W), dtype=np.float32)
    fire = (u_fire < arr_c)

    # --- Latencias: t = floor((1 - I)*(T-1)), clamp a [0, T-1] ---
    t_float = (1.0 - arr_c) * (T - 1)
    t_idx = np.floor(t_float).astype(np.int64)
    np.clip(t_idx, 0, T - 1, out=t_idx)

    # --- Escribir spikes one-hot temporal ---
    for c in range(C):
        yy, xx = np.where(fire[c])
        if yy.size:
            tt = t_idx[c, yy, xx]
            spikes[tt, c, yy, xx] = 1.0

    # --- Salida con/sin canal ---
    if img01.ndim == 2:
        return spikes[:, 0, :, :]  # (T,H,W)
    else:
        return spikes              # (T,C,H,W)


def encode_csv_to_h5(
    csv_df_or_path: pd.DataFrame | Path | str,
    base_dir: Path,
    out_path: Path,
    *,
    encoder: str = "rate",     # "rate" | "latency" | "raw"
    T: int = 20,
    gain: float = 0.5,
    size_wh: tuple[int, int] = (200, 66),
    to_gray: bool = True,
    crop_top: int = 0,
    crop_bottom: int = 0,
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
    # Detecta carpeta processed/<run> a partir del CSV si es una ruta de archivo
    if isinstance(csv_df_or_path, (str, Path)):
        proc_dir_for_csv = Path(csv_df_or_path).parent
    else:
        # Si vino un DataFrame, asumimos 'base_dir' como raw y usamos el mismo como fallback de proc
        proc_dir_for_csv = base_dir

    first = _imread_center_mixed(
        df.iloc[0],
        base_raw=base_dir,
        base_proc=proc_dir_for_csv,
        to_gray=to_gray,
        size_wh=(W, H),
        crop_top=crop_top,
        crop_bottom=crop_bottom,
    )

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
    with h5py.File(out_path, "w", locking=False) as h5:
        # Atributos
        h5.attrs["version"] = 2
        h5.attrs["encoder"] = encoder
        h5.attrs["T"] = int(T)
        h5.attrs["gain"] = float(gain)
        h5.attrs["size_wh"] = np.asarray([W, H], dtype=np.int32)
        h5.attrs["to_gray"] = int(bool(to_gray))
        h5.attrs["crop_top"] = int(crop_top)
        h5.attrs["crop_bottom"] = int(crop_bottom)
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
            img01 = _imread_center_mixed(
                row,
                base_raw=base_dir,
                base_proc=proc_dir_for_csv,
                to_gray=to_gray,
                size_wh=(W, H),
                crop_top=crop_top,
                crop_bottom=crop_bottom,
            )
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

    # === Manifest (upsert) ====================================================
    # Clave de manifest: el nombre del archivo H5 en ese directorio
    key = out_path.name
    # CSV de origen como string legible
    if isinstance(csv_df_or_path, (str, Path)):
        src_csv_str = str(csv_df_or_path)
    else:
        src_csv_str = "<DataFrame>"

    entry = {
        "out_path": key,                 # clave == nombre del H5
        "version": 2,                    # coincide con h5.attrs["version"]
        "encoder": str(encoder),
        "T": int(T),
        "gain": float(gain) if encoder == "rate" else 0.0,
        "size_wh": [int(W), int(H)],     # W,H (coherente con el resto del repo)
        "to_gray": bool(to_gray),
        "crop_top": int(crop_top),
        "crop_bottom": int(crop_bottom),
        "channels": int(channels),
        "seed": int(seed),
        "compression": str(compression),
        "n_samples": int(N),
        "source_csv": src_csv_str,
        "base_dir_raw": str(base_dir),
        "base_dir_proc": str(proc_dir_for_csv),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _upsert_prep_manifest(out_path.parent / "prep_manifest.json", key=key, entry=entry)

