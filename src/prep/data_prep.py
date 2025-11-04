# -*- coding: utf-8 -*-
"""Preparación offline del dataset Udacity:

- Limpieza de rutas (normaliza y las deja relativas a IMG/).
- Filtro de filas cuya imagen no exista.
- (Opcional) Expansión de cámaras: genera filas nuevas con 'center' apuntando
  a la imagen 'left' o 'right' y corrige el `steering` en ±shift (clip [-1,1]).
- Split estratificado por bins de 'steering' → train/val/test.
- (Opcional) Oversampling por bins en train:
    * Modo LEGACY: duplica filas → train_balanced.csv.
    * NUEVO MODO 'images': genera imágenes augmentadas bajo processed/<run>/aug/train/IMG
      y crea train_balanced.csv apuntando a esas rutas (prefijo 'aug/...').
- Escritura de:
    * canonical.csv (limpio sin balanceo)
    * train.csv / val.csv / test.csv
    * train_balanced.csv (si se pidió balanceo)
    * tasks.json / tasks_balanced.json
    * prep_manifest.json (resumen reproducible de la preparación)

NOTA DE COMPATIBILIDAD:
- El entrenamiento usa `UdacityCSV` que ahora entiende:
  * 'IMG/...' → data/raw/udacity/<run> (raw)
  * 'aug/...' → data/processed/<run>    (processed)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Tuple

import argparse, json
import numpy as np
import pandas as pd

# ------------------------------- Utils --------------------------------
def _json_default(o):
    # Paths ⇒ str
    if isinstance(o, Path):
        return str(o)
    # numpy escalares ⇒ tipos Python
    try:
        import numpy as _np
        if isinstance(o, (_np.integer, _np.floating)):
            return o.item()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _fix_path(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).strip().strip('"').strip("'").replace("\\", "/")
    return s


def _rel_to_img(s: str) -> str:
    """Devuelve ruta relativa desde 'IMG/...' si aparece; si no, la deja igual."""
    if pd.isna(s):
        return s
    low = s.lower()
    k = low.rfind("img/")
    return s[k:] if k != -1 else s


def _clip_steer(x: float) -> float:
    return float(np.clip(float(x), -1.0, 1.0))


# --------------------------- Carga y limpieza --------------------------
CSV_COLS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def load_raw_log(base: Path) -> pd.DataFrame:
    """Lee driving_log.csv, normaliza rutas y filtra filas sin imágenes."""
    csv = base / "driving_log.csv"
    assert csv.exists(), f"No existe {csv}"
    df = pd.read_csv(csv, header=None, names=CSV_COLS)

    # Normaliza y relativiza
    for cam in ["center", "left", "right"]:
        df[cam] = df[cam].map(_fix_path).map(_rel_to_img)

    # Convierte numéricos
    for c in ["steering", "throttle", "brake", "speed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filtra filas sin imagen (al menos center)
    df = df[df["center"].map(lambda s: (base / s).exists())]
    df = df.dropna(subset=["steering"]).reset_index(drop=True)
    return df


def _rel_to_run_from_log(s: str, log_dir: Path, run_root: Path) -> str:
    """Normaliza s y la convierte a ruta RELATIVA a run_root (p.ej. 'vuelta1/IMG/...')."""
    if pd.isna(s):
        return s
    s = _fix_path(s)

    low = s.lower()
    k = low.rfind("img/")
    if k != -1:
        abs_p = (log_dir / s[k:]).resolve()
    else:
        p = Path(s)
        abs_p = p if p.is_absolute() else (log_dir / p)
        abs_p = abs_p.resolve()

    # Intentar relativo a run_root
    try:
        rel = abs_p.relative_to(run_root.resolve())
        return rel.as_posix()
    except Exception:
        low2 = str(abs_p).lower()
        k2 = low2.rfind("img/")
        if k2 != -1:
            return str(abs_p)[k2:].replace("\\", "/")
        return abs_p.name


def load_raw_logs_merged(run_root: Path) -> pd.DataFrame:
    """Fusiona TODOS los driving_log.csv bajo run_root (recursivo), ignorando 'aug/'."""
    logs = [p for p in run_root.rglob("driving_log.csv") if "aug" not in p.parts]
    if not logs:
        raise FileNotFoundError(f"No se encontraron driving_log.csv bajo {run_root}")
    dfs = []
    for lp in sorted(logs):
        df = pd.read_csv(lp, header=None, names=CSV_COLS)
        for cam in ["center", "left", "right"]:
            if cam in df.columns:
                df[cam] = df[cam].map(lambda s: _rel_to_run_from_log(s, lp.parent, run_root))
        for c in ["steering", "throttle", "brake", "speed"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[df["center"].map(lambda s: (run_root / s).exists())]
        df = df.dropna(subset=["steering"]).reset_index(drop=True)
        dfs.append(df)
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        raise RuntimeError(
            f"Todas las filas fueron filtradas por inexistentes en {run_root}. "
            f"¿Rutas del CSV no casan con el árbol de imágenes?"
        )
    return pd.concat(dfs, ignore_index=True)


# ----------------------- Expansión de cámaras -------------------------
def expand_cameras_into_center(df: pd.DataFrame, steer_shift: float) -> pd.DataFrame:
    """Duplica filas usando left/right en 'center' con corrección del steering."""
    base = df.copy()
    left_df = df.copy()
    left_df["center"] = df["left"]
    left_df["steering"] = (df["steering"].astype(float) + float(steer_shift)).map(_clip_steer)

    right_df = df.copy()
    right_df["center"] = df["right"]
    right_df["steering"] = (df["steering"].astype(float) - float(steer_shift)).map(_clip_steer)

    out = pd.concat([base, left_df, right_df], ignore_index=True)
    return out


# ---------------------- Split estratificado por bins ------------------
def stratified_split(
    df: pd.DataFrame,
    bins: int = 21,
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
    return_edges: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Split estratificado por histogramas de `steering` (pd.cut)."""
    s = df["steering"].astype(float).clip(-1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, bins)
    lab = pd.cut(s, bins=edges, include_lowest=True, labels=False)

    rng = np.random.default_rng(seed)
    parts = []
    for b in sorted(lab.dropna().unique()):
        g = df[lab == b]
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(n * train))
        nva = int(round(n * val))
        parts.append((g.iloc[idx[:ntr]], g.iloc[idx[ntr:ntr + nva]], g.iloc[idx[ntr + nva:]]))

    tr = pd.concat([a for a, _, _ in parts], ignore_index=True)
    va = pd.concat([b for _, b, _ in parts], ignore_index=True)
    te = pd.concat([c for _, _, c in parts], ignore_index=True)
    if return_edges:
        return tr, va, te, edges
    return tr, va, te


def verify_processed_splits(proc_root: Path, runs: list[str]) -> None:
    missing = []
    for run in runs:
        base = proc_root / run
        for part in ["train", "val", "test"]:
            p = base / f"{part}.csv"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError("Faltan CSV obligatorios:\n" + "\n".join(" - " + m for m in missing))


# -------------------------- Oversampling por bins (LEGACY) ---------------------
def make_balanced_by_bins(
    train_df: pd.DataFrame,
    bins: int = 21,
    target_per_bin: str | int = "auto",
    cap_per_bin: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Oversampling por bins duplicando filas (LEGACY)."""
    s = train_df["steering"].astype(float).clip(-1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, bins)
    lab = pd.cut(s, bins=edges, include_lowest=True, labels=False)

    counts = lab.value_counts().sort_index()
    max_bin = int(counts.max()) if len(counts) else 0
    if isinstance(target_per_bin, str) and target_per_bin == "auto":
        target = max_bin
        if cap_per_bin is not None:
            target = min(target, int(cap_per_bin))
    else:
        target = int(target_per_bin)

    rng = np.random.default_rng(seed)
    out_parts = []
    for b in range(len(edges) - 1):
        g = train_df[lab == b]
        if g.empty:
            continue
        k = target - len(g)
        if k <= 0:
            out_parts.append(g)
        else:
            idx = rng.integers(low=0, high=len(g), size=k)
            dup = g.iloc[idx]
            out_parts.append(pd.concat([g, dup], ignore_index=True))

    balanced = pd.concat(out_parts, ignore_index=True) if out_parts else train_df.copy()
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


# -------------------------- Balanceo OFFLINE por imágenes ----------------------
# Esta sección crea físicamente imágenes augmentadas en processed/<run>/aug/train/IMG
# y devuelve un train_balanced DataFrame con rutas 'aug/train/IMG/...'.
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_bgr_from_raw(center_rel: str, base_raw: Path) -> np.ndarray:
    p = (base_raw / center_rel).resolve()
    img = cv2.imread(str(p), cv2.IMREAD_COLOR) if _HAS_CV2 else None
    if img is None:
        raise FileNotFoundError(f"No se pudo leer (raw): {p}")
    return img


def _apply_offline_aug_bgr(bgr: np.ndarray,
                           brightness: Tuple[float, float] | None,
                           contrast:   Tuple[float, float] | None,
                           saturation: Tuple[float, float] | None,
                           hue:        Tuple[float, float] | None,
                           gamma:      Tuple[float, float] | None,
                           noise_std:  float,
                           prob_hflip: float,
                           rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Devuelve (bgr_aug, flipped). Trabaja en uint8 BGR por compatibilidad OpenCV."""
    img = bgr.astype(np.float32) / 255.0
    flipped = False

    # brillo
    if brightness is not None:
        lo, hi = brightness
        alpha = float(rng.uniform(float(lo), float(hi)))
        img = np.clip(img * alpha, 0.0, 1.0)

    # contraste
    if contrast is not None:
        lo, hi = contrast
        c = float(rng.uniform(float(lo), float(hi)))
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * c + mean, 0.0, 1.0)

    # saturación/hue en HSV
    if saturation is not None or hue is not None:
        hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if saturation is not None:
            lo, hi = saturation
            s = float(rng.uniform(float(lo), float(hi)))
            S = np.clip(S * s, 0, 255)
        if hue is not None:
            lo, hi = hue
            # Interpretamos radianes aprox → escala a grados (≈ rad*180/pi, ~57.3)
            delta_deg = float(rng.uniform(float(lo), float(hi))) * 57.2957795
            H = (H + delta_deg) % 180.0
        hsv2 = np.stack([H, S, V], axis=-1).astype(np.uint8)
        img = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # gamma
    if gamma is not None:
        g_lo, g_hi = gamma
        g = max(1e-6, float(rng.uniform(float(g_lo), float(g_hi))))
        img = np.clip(np.power(img, g), 0.0, 1.0)

    # ruido
    if noise_std and noise_std > 0:
        noise = rng.normal(0.0, float(noise_std), size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    # flip
    if prob_hflip > 0 and rng.random() < prob_hflip:
        img = img[:, ::-1, :]
        flipped = True

    out = (img * 255.0).astype(np.uint8)
    return out, flipped


def make_balanced_images_offline(
    train_df: pd.DataFrame,
    base_raw: Path,
    base_proc: Path,
    bins: int,
    target_per_bin: Union[str, int],
    cap_per_bin: Optional[int],
    aug_cfg: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Crea imágenes augmentadas en processed/<run>/aug/train/IMG y devuelve
    un DataFrame train_balanced (train + aug)."""
    rng = np.random.default_rng(seed)

    s = train_df["steering"].astype(float).clip(-1.0, 1.0)
    edges = np.linspace(-1.0, 1.0, int(bins))
    lab = pd.cut(s, bins=edges, include_lowest=True, labels=False)

    counts = lab.value_counts().sort_index()
    max_bin = int(counts.max()) if len(counts) else 0
    if isinstance(target_per_bin, str) and target_per_bin == "auto":
        target = max_bin
        if cap_per_bin is not None:
            target = min(target, int(cap_per_bin))
    else:
        target = int(target_per_bin)

    out_rows = []
    aug_root = (base_proc / "aug" / "train" / "IMG").resolve()
    _ensure_dir(aug_root)

    # Parametrización de aumentos
    prob_hflip = float(aug_cfg.get("prob_hflip", 0.0))
    brightness = tuple(aug_cfg.get("brightness")) if aug_cfg.get("brightness") else None
    contrast   = tuple(aug_cfg.get("contrast"))   if aug_cfg.get("contrast")   else None
    saturation = tuple(aug_cfg.get("saturation")) if aug_cfg.get("saturation") else None
    hue        = tuple(aug_cfg.get("hue"))        if aug_cfg.get("hue")        else None
    gamma      = tuple(aug_cfg.get("gamma"))      if aug_cfg.get("gamma")      else None
    noise_std  = float(aug_cfg.get("noise_std", 0.0))

    # Generar por-bin
    for b in range(len(edges) - 1):
        g = train_df[lab == b]
        if g.empty:
            continue
        k = target - len(g)
        if k <= 0:
            continue
        idx = rng.integers(low=0, high=len(g), size=k)
        dup = g.iloc[idx].copy()

        for i, row in dup.iterrows():
            center_rel = str(row["center"]).replace("\\", "/")
            # Leer original desde RAW
            bgr = _read_bgr_from_raw(center_rel, base_raw)
            # Augment
            bgr_aug, flipped = _apply_offline_aug_bgr(
                bgr, brightness, contrast, saturation, hue, gamma, noise_std, prob_hflip, rng
            )
            # Nombre único
            stem = Path(center_rel).name
            out_name = f"{Path(stem).stem}__aug_{rng.integers(1_000_000_000)}.jpg"
            out_path = aug_root / out_name
            cv2.imwrite(str(out_path), bgr_aug)

            # Nueva fila: center -> 'aug/train/IMG/<out_name>'
            new_row = row.copy()
            new_row["center"] = f"aug/train/IMG/{out_name}"
            if flipped:
                try:
                    new_row["steering"] = -float(new_row["steering"])
                except Exception:
                    pass
            out_rows.append(new_row)

    if not out_rows:
        # Si no se ha creado nada, devolvemos original barajado
        return train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    aug_df = pd.DataFrame(out_rows)
    balanced = pd.concat([train_df, aug_df], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


# --------------------------- Escritura salidas ------------------------
def _write_splits(out_dir: Path, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.csv").write_text("", encoding="utf-8")  # pre-crea por atomicidad
    tr.to_csv(out_dir / "train.csv", index=False)
    va.to_csv(out_dir / "val.csv", index=False)
    te.to_csv(out_dir / "test.csv", index=False)


def _write_tasks_json(proc_root: Path, runs: list[str], balanced: bool):
    obj = {"tasks_order": runs, "splits": {}}
    name = "tasks_balanced.json" if balanced else "tasks.json"
    for run in runs:
        base = proc_root / run
        tr = base / ("train_balanced.csv" if balanced else "train.csv")
        obj["splits"][run] = {
            "train": str((tr).resolve()),
            "val":   str((base / "val.csv").resolve()),
            "test":  str((base / "test.csv").resolve()),
        }
    (proc_root / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return proc_root / name


# ------------------------------- Main ---------------------------------
@dataclass
class PrepConfig:
    root: Path
    runs: list[str]
    use_left_right: bool
    steer_shift: float
    bins: int
    train: float
    val: float
    seed: int
    target_per_bin: Optional[Union[str, int]] = None   # LEGACY duplication mode
    cap_per_bin: Optional[int] = None
    merge_subruns: bool = False

    # Nuevo bloque de balanceo offline por imágenes (processed/aug)
    # Si está presente en el preset como:
    #   balance_offline:
    #       mode: images | none
    #       target_per_bin: auto | <int>
    #       cap_per_bin: 12000
    #       aug: {... mismos campos que aug_train ...}
    balance_offline: Optional[dict] = None


def run_prep(cfg: PrepConfig) -> dict:
    RAW  = cfg.root / "data" / "raw" / "udacity"
    PROC = cfg.root / "data" / "processed"
    PROC.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": asdict(cfg),
        "runs": {},
    }
    for run in cfg.runs:
        base = RAW / run                   # raw/<run>
        out_dir = PROC / run               # processed/<run>
        out_dir.mkdir(parents=True, exist_ok=True)

        df = load_raw_logs_merged(base) if cfg.merge_subruns else load_raw_log(base)
        df.to_csv(out_dir / "canonical.csv", index=False)

        if cfg.use_left_right:
            df = expand_cameras_into_center(df, cfg.steer_shift)

        tr, va, te = stratified_split(df, bins=cfg.bins, train=cfg.train, val=cfg.val, seed=cfg.seed)
        _write_splits(out_dir, tr, va, te)

        # --- Balanceo por IMÁGENES offline (nuevo flujo) ---
        trb_path = None
        if cfg.balance_offline and str(cfg.balance_offline.get("mode", "none")).lower() == "images":
            target_pb = cfg.balance_offline.get("target_per_bin", "auto")
            cap_pb    = cfg.balance_offline.get("cap_per_bin", None)
            aug_cfg   = cfg.balance_offline.get("aug", {}) or {}

            tr_balanced = make_balanced_images_offline(
                tr,
                base_raw=base,
                base_proc=out_dir,
                bins=cfg.bins,
                target_per_bin=target_pb,
                cap_per_bin=cap_pb,
                aug_cfg=aug_cfg,
                seed=cfg.seed,
            )
            if len(tr_balanced) > len(tr):
                trb_path = out_dir / "train_balanced.csv"
                tr_balanced.to_csv(trb_path, index=False)

        # --- Alternativa LEGACY: duplicación de filas (si se ha configurado target_per_bin) ---
        elif cfg.target_per_bin not in (None, "none", ""):
            tr_balanced = make_balanced_by_bins(
                tr,
                bins=cfg.bins,
                target_per_bin=cfg.target_per_bin,
                cap_per_bin=cfg.cap_per_bin,
                seed=cfg.seed
            )
            if len(tr_balanced) > len(tr):
                trb_path = out_dir / "train_balanced.csv"
                tr_balanced.to_csv(trb_path, index=False)

        manifest["runs"][run] = {
            "canonical": str(out_dir / "canonical.csv"),
            "train":     str(out_dir / "train.csv"),
            "val":       str(out_dir / "val.csv"),
            "test":      str(out_dir / "test.csv"),
            "train_balanced": str(trb_path) if trb_path else None,
            "stats": {
                "n_canonical": int(len(pd.read_csv(out_dir / "canonical.csv"))),
                "n_train":     int(len(tr)),
                "n_val":       int(len(va)),
                "n_test":      int(len(te)),
                "n_train_balanced": (int(len(pd.read_csv(trb_path))) if trb_path else None),
            }
        }

    tasks_path = _write_tasks_json(PROC, cfg.runs, balanced=False)
    balanced_exists = all((PROC / run / "train_balanced.csv").exists() for run in cfg.runs)
    tasks_balanced_path = _write_tasks_json(PROC, cfg.runs, balanced=True) if balanced_exists else None

    manifest["outputs"] = {
        "tasks_json": str(tasks_path),
        "tasks_balanced_json": str(tasks_balanced_path) if tasks_balanced_path else None,
    }
    (PROC / "prep_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8"
    )
    return manifest