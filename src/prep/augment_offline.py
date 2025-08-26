# src/prep/augment_offline.py
from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _apply_augment_bgr(img_bgr_uint8: np.ndarray, rng: np.random.Generator, aug: dict) -> np.ndarray:
    """Aplica augment en BGR uint8 (0..255). Devuelve uint8 BGR."""
    img = img_bgr_uint8.astype(np.float32) / 255.0

    # Horizontal flip
    p = float(aug.get("prob_hflip", 0.0) or 0.0)
    if p > 0.0 and rng.random() < p:
        img = img[:, ::-1, :]

    # Brillo
    br = aug.get("brightness", None)
    if br is not None:
        f = float(rng.uniform(br[0], br[1]))
        img = _clamp01(img * f)

    # Contraste
    ct = aug.get("contrast", None)
    if ct is not None:
        c = float(rng.uniform(ct[0], ct[1]))
        mean = img.mean(axis=(0,1), keepdims=True)
        img = _clamp01((img - mean) * c + mean)

    # Saturación / Tinte (HSV)
    sat = aug.get("saturation", None)
    hue = aug.get("hue", None)
    if sat is not None or hue is not None:
        hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)
        if sat is not None:
            s_factor = float(rng.uniform(sat[0], sat[1]))
            S = np.clip(S * s_factor, 0, 255)
        if hue is not None:
            shift = float(rng.uniform(hue[0], hue[1])) * 179.0
            H = (H + shift) % 180.0
        hsv = cv2.merge([H, S, V]).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Gamma
    gm = aug.get("gamma", None)
    if gm is not None:
        gamma = float(rng.uniform(gm[0], gm[1]))
        img = _clamp01(np.power(img, gamma))

    # Ruido (ns en escala 0..255)
    ns = float(aug.get("noise_std", 0.0) or 0.0)
    if ns > 0.0:
        noise = rng.normal(0.0, ns/255.0, size=img.shape).astype(np.float32)
        img = _clamp01(img + noise)

    return (img * 255.0).astype(np.uint8)

def balance_train_with_augmented_images(
    train_csv: Path,
    raw_run_dir: Path,
    out_run_dir: Path,
    *,
    bins: int,
    target_per_bin: int | str = "auto",
    cap_per_bin: int | None = None,
    seed: int = 42,
    aug: dict | None = None,
    idempotent: bool = True,         # ← NUEVO
    overwrite: bool = False          # ← NUEVO
) -> tuple[Path, dict]:
    """
    Genera imágenes aumentadas reales para equilibrar bins de 'steering' en TRAIN.
    Idempotente si `idempotent=True` (por defecto). Usa un manifest con la config.
    """
    import shutil, json

    out_run_dir = Path(out_run_dir)
    out_aug_dir = out_run_dir / "aug" / "train"
    out_aug_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_run_dir / "train_balanced.csv"
    manifest_path = out_aug_dir / "_aug_manifest.json"

    # --- Config "deseada" (para comparar) ---
    desired_cfg = {
        "bins": int(bins),
        "target_per_bin": target_per_bin,
        "cap_per_bin": cap_per_bin,
        "seed": int(seed),
        "aug": (aug or {}),
        "source_train_csv": str(train_csv)
    }

    # --- Atajo idempotente ---
    if idempotent and out_csv.exists() and manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prev.get("config") == desired_cfg:
            # misma config ⇒ no regeneramos
            print("[augment_offline] Balanceo ya existente con la misma configuración; no se regenera.")
            return out_csv, prev.get("stats", {})

        if not overwrite:
            print("[augment_offline] Configuración distinta a la previa; conserva lo existente. "
                  "Pasa overwrite=True para regenerar.")
            return out_csv, prev.get("stats", {})

        # overwrite=True ⇒ limpiamos y seguimos
        print("[augment_offline] Config distinta: purgando carpeta de augment y regenerando…")
        try:
            shutil.rmtree(out_aug_dir)
        except Exception:
            pass
        out_aug_dir.mkdir(parents=True, exist_ok=True)

    # === A partir de aquí, generación normal ===
    df = pd.read_csv(train_csv)
    assert "center" in df.columns and "steering" in df.columns, "CSV debe tener 'center' y 'steering'."

    s = df["steering"].astype(float).clip(-1.0, 1.0).to_numpy()
    edges = np.linspace(-1.0, 1.0, int(bins))
    lab = pd.cut(s, bins=edges, include_lowest=True, labels=False)

    counts = pd.Series(lab).value_counts().sort_index()
    max_bin = int(counts.max()) if len(counts) else 0
    if isinstance(target_per_bin, str) and target_per_bin == "auto":
        target = max_bin if cap_per_bin is None else min(max_bin, int(cap_per_bin))
    else:
        target = int(target_per_bin)

    rng = _rng(seed)
    aug = aug or {}
    parts = [df]  # originales
    generated = 0

    for b in range(len(edges) - 1):
        g = df[lab == b]
        if g.empty:
            continue
        need = target - len(g)
        if need <= 0:
            continue

        idx = rng.integers(low=0, high=len(g), size=need)
        to_aug = g.iloc[idx].reset_index(drop=True)

        new_rows = []
        for i, row in to_aug.iterrows():
            src_rel = str(row["center"])
            src_abs = (raw_run_dir / src_rel).resolve()

            img_bgr = cv2.imread(str(src_abs), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            img_aug = _apply_augment_bgr(img_bgr, rng, aug)

            stem = Path(src_rel).stem
            out_name = f"{stem}_aug_b{b}_{i}_{rng.integers(10**9)}.jpg"
            out_abs = out_aug_dir / out_name
            out_abs.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_abs), img_aug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            r = row.copy()
            r["center"] = f"aug/train/{out_name}"   # relativo a processed/<run>
            new_rows.append(r)

        if new_rows:
            parts.append(pd.DataFrame(new_rows))
            generated += len(new_rows)

    out_df = pd.concat(parts, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)

    stats = {
        "bins": int(bins),
        "target_per_bin": target_per_bin,
        "cap_per_bin": cap_per_bin,
        "generated": int(generated),
        "out_csv": str(out_csv),
        "out_aug_dir": str(out_aug_dir),
    }

    # Guardar manifest para idempotencia
    manifest_path.write_text(json.dumps({"config": desired_cfg, "stats": stats}, indent=2), encoding="utf-8")
    return out_csv, stats

