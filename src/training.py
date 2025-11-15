# src/training.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import set_seeds
# Helpers neutrales (y RE-EXPORT para compatibilidad con imports antiguos)
from .nn_io import set_encode_runtime, _align_target_shape, _forward_with_cached_orientation
__all__ = [
    "TrainConfig",
    "train_supervised",
    # re-export helpers (compatibilidad con código existente)
    "set_encode_runtime",
    "_align_target_shape",
    "_forward_with_cached_orientation",
]

# Rendimiento numérico en Ada: TF32 ON
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ── NUEVO: utilidades de memoria/log
def _mem_snapshot():
    # CPU
    rss_mb = None
    try:
        import psutil
        p = psutil.Process(os.getpid())
        rss_mb = float(p.memory_info().rss) / (1024 ** 2)
    except Exception:
        try:
            import resource
            rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # en Linux ru_maxrss ya viene en KB, en macOS es en bytes -> nos vale como aproximación
            rss_mb = rss_kb / 1024.0
        except Exception:
            rss_mb = None

    # CUDA
    cuda = {"alloc_mb": None, "resvd_mb": None, "peak_mb": None}
    if torch.cuda.is_available():
        try:
            stats = torch.cuda.memory_stats()
            cuda["alloc_mb"] = stats["allocated_bytes.all.current"] / (1024 ** 2)
            cuda["resvd_mb"] = stats["reserved_bytes.all.current"] / (1024 ** 2)
            cuda["peak_mb"]  = stats["allocated_bytes.all.peak"] / (1024 ** 2)
        except Exception:
            pass
    return {"rss_mb": rss_mb, **cuda}

def _maybe_log_mem(it_count: int, every: int):
    if every <= 0:
        return
    if (it_count % every) != 0:
        return
    snap = _mem_snapshot()
    print(
        f"[MEM] rss={snap['rss_mb']:.0f}MB | "
        f"cuda_alloc={'' if snap['alloc_mb'] is None else f'{snap['alloc_mb']:.0f}MB'} | "
        f"cuda_resvd={'' if snap['resvd_mb'] is None else f'{snap['resvd_mb']:.0f}MB'} | "
        f"cuda_peak={'' if snap['peak_mb'] is None else f'{snap['peak_mb']:.0f}MB'}"
    )

@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    amp: bool = True
    seed: Optional[int] = None
    es_patience: Optional[int] = None
    es_min_delta: float = 0.0

def train_supervised(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    out_dir: Path,
    method=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = out_dir / "best.pth"
    last_ckpt = out_dir / "last.pth"

    if cfg.seed is not None:
        set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_amp = bool(cfg.amp and torch.cuda.is_available())
    # <<< CAMBIO CLAVE: usar GradScaler solo en FP16; BF16 no escala >>>
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    dtype_autocast = (torch.bfloat16 if use_bf16 else torch.float16) if use_amp else None

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=(use_amp and not use_bf16))  # solo FP16

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_mse": []}
    t0 = time.time()
    best_val = float("inf")
    patience_left = cfg.es_patience if (cfg.es_patience is not None and cfg.es_patience > 0) else None
    best_state = None

    LOG_ITPS = os.environ.get("TRAIN_LOG_ITPS", "0") == "1"
    LOG_MEM = os.environ.get("TRAIN_LOG_MEM", "0") == "1"
    LOG_MEM_EVERY = int(os.environ.get("TRAIN_LOG_MEM_EVERY", "64"))

    phase_hint: Dict[str, str] = {"train": None, "val": None}

    # Permite que el método envuelva el loader si lo necesita (p.ej., Rehearsal)
    if method is not None and hasattr(method, "prepare_train_loader"):
        try:
            maybe_loader = method.prepare_train_loader(train_loader)
            if maybe_loader is not None:
                train_loader = maybe_loader
        except Exception:
            # Caída blanda: si algo falla aquí, sigue con el loader original
            pass

    for epoch in range(1, cfg.epochs + 1):
        if method is not None and hasattr(method, "before_epoch"):
            try:
                method.before_epoch(model, epoch)
            except Exception:
                pass

        model.train()
        running = 0.0
        it_count = 0
        t_epoch0 = time.perf_counter()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            # Mueve y a device (x lo mueve _forward_* si es necesario)
            y = y.to(device, non_blocking=True)

            # <<< autocast (BF16 si disponible) cubre forward+loss; sin AMP interno >>>
            with autocast(device_type="cuda", enabled=use_amp, dtype=dtype_autocast):
                y_hat = _forward_with_cached_orientation(
                    model=model, x=x, y=y, device=device,
                    use_amp=False,  # << evitamos doble autocast; usamos el nuestro
                    phase_hint=phase_hint, phase="train"
                )
                # Alinea target y garantiza mismo dtype que y_hat
                y_aligned = _align_target_shape(y_hat, y).to(
                    device=y_hat.device, dtype=y_hat.dtype, non_blocking=True
                )
                loss_base = loss_fn(y_hat, y_aligned)

                # Regularización del método al mismo dtype
                pen = 0.0
                if method is not None:
                    p = method.penalty()
                    if isinstance(p, torch.Tensor):
                        pen = p.to(device=y_hat.device, dtype=y_hat.dtype)
                    else:
                        pen = torch.tensor(float(p), device=y_hat.device, dtype=y_hat.dtype)
                loss = loss_base + pen

            # Logging interno opcional
            log_inner = bool(getattr(method, "inner_verbose", False))
            log_every = int(getattr(method, "inner_every", 50))
            if method is not None and log_inner and (it_count % max(1, log_every) == 0):
                base_val = float(loss_base.detach().item())
                pen_val = float(pen.detach().item() if isinstance(pen, torch.Tensor) else float(pen))
                ratio = pen_val / max(1e-8, base_val)
                suggest_msg = ""
                tun = getattr(method, "tunable", None)
                if callable(tun):
                    try:
                        info = tun() or {}
                        if info.get("strategy") == "ratio":
                            curr = float(info.get("value", 0.0))
                            target = float(info.get("target_ratio", 1.0))
                            param = str(info.get("param", "λ"))
                            if curr > 0.0 and base_val > 0.0 and pen_val > 0.0:
                                next_val = curr * (target / max(1e-8, ratio))
                                suggest_msg = f" | {param}_actual={curr:.3e} → sugerido≈{next_val:.3e} (target pen/base={target})"
                    except Exception:
                        pass
                meth = str(getattr(method, "name", "method"))
                print(f"[{meth}] base={base_val:.4g} | pen={pen_val:.4g} | pen/base={ratio:.3f}{suggest_msg}")

            if method is not None and hasattr(method, "before_batch"):
                try:
                    method.before_batch(model, (x, y))
                except Exception:
                    pass

            opt.zero_grad(set_to_none=True)
            if use_amp and not use_bf16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            running += float(loss.detach().item())

            if method is not None and hasattr(method, "after_batch"):
                try:
                    method.after_batch(model, (x, y), loss)
                except Exception:
                    pass

            it_count += 1
            if LOG_MEM:
                _maybe_log_mem(it_count, LOG_MEM_EVERY)

        if LOG_ITPS and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_epoch0
        if LOG_ITPS and dt > 0:
            ips = it_count / dt
            print(f"[TRAIN it/s] epoch {epoch}/{cfg.epochs}: {ips:.1f} it/s ({it_count} iters en {dt:.2f}s)")

        train_loss = running / max(1, len(train_loader))

        # ------------------- Validación (MSE + MAE) -------------------
        model.eval()
        v_running_mse = 0.0
        v_running_mae = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                y = y.to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=use_amp, dtype=dtype_autocast):
                    y_hat = _forward_with_cached_orientation(
                        model=model, x=x, y=y, device=device,
                        use_amp=False,  # sin doble autocast en val
                        phase_hint=phase_hint, phase="val"
                    )
                    y_aligned = _align_target_shape(y_hat, y).to(
                        device=y_hat.device, dtype=y_hat.dtype, non_blocking=True
                    )
                    v_loss = loss_fn(y_hat, y_aligned)

                v_running_mse += float(v_loss.detach().item())
                mae_batch = torch.abs(y_hat.to(torch.float32) - y_aligned.to(torch.float32)).mean()
                v_running_mae += float(mae_batch.detach().item())
                n_val_batches += 1

        val_loss = v_running_mse / max(1, n_val_batches)
        val_mae = v_running_mae / max(1, n_val_batches)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_mse"].append(val_loss)

        if method is not None and hasattr(method, "after_epoch"):
            try:
                method.after_epoch(model, epoch)
            except Exception:
                pass

        torch.save(model.state_dict(), last_ckpt)

        if patience_left is not None:
            improved = (best_val - val_loss) > cfg.es_min_delta
            if improved:
                best_val = val_loss
                patience_left = cfg.es_patience
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_ckpt)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    elapsed = time.time() - t0

    if (not best_ckpt.exists()) and last_ckpt.exists():
        try:
            sd = torch.load(last_ckpt, map_location="cpu")
            torch.save(sd, best_ckpt)
        except Exception:
            pass

    manifest = {
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "amp": cfg.amp,
        "seed": cfg.seed,
        "elapsed_sec": elapsed,
        "device": str(device),
        "history": history,
        "early_stopping": {
            "used": bool(cfg.es_patience and cfg.es_patience > 0),
            "patience": cfg.es_patience,
            "min_delta": cfg.es_min_delta,
            "best_val": best_val if best_val != float("inf") else None,
        },
        "checkpoints": {"best": str(best_ckpt), "last": str(last_ckpt)},
        "mem_last": _mem_snapshot(),  # NUEVO
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Dump opcional de memoria CUDA al final (si se pide)
    if os.environ.get("TRAIN_DUMP_MEM_SUMMARY", "0") == "1" and torch.cuda.is_available():
        try:
            (out_dir / "memory_summary.txt").write_text(
                torch.cuda.memory_summary(device=None, abbreviated=False),
                encoding="utf-8"
            )
        except Exception:
            pass

    return history
