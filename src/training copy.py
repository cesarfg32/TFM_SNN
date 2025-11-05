# src/training.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json, time, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import torch
from torch import nn, optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import set_seeds  # reproducibilidad global

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Para resultados finales reproducibles mejor dejarlo en False;
# si quieres exprimir velocidad durante HPO, puedes poner True:
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Runtime encode: permite que el DataLoader entregue 4D (B,C,H,W)
# y aquí hacemos la codificación temporal en GPU (T,B,C,H,W).
# ---------------------------------------------------------------------
_RUNTIME_ENC = {"mode": None, "T": None, "gain": None, "device": torch.device("cpu")}

def set_encode_runtime(mode: str | None, T: int | None = None, gain: float | None = None, device=None):
    """
    Activa/desactiva codificación temporal en runtime para inputs 4D (B,C,H,W).
    - mode: "rate" | "latency" | "raw" | None
    - T, gain: parámetros del encoder
    - device: torch.device destino (por defecto CUDA si disponible)
    """
    global _RUNTIME_ENC
    if mode is None:
        _RUNTIME_ENC.update({"mode": None, "T": None, "gain": None, "device": torch.device("cpu")})
        return
    _RUNTIME_ENC.update({
        "mode": str(mode),
        "T": int(T) if T is not None else None,
        "gain": float(gain) if gain is not None else None,
        "device": device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
    })


def _align_target_shape(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Alinea la forma del target con la predicción:
    - Si y_hat=(B,1) y y=(B,) -> y=(B,1)
    - Si y_hat=(B,)   y y=(B,1) -> y=(B,)
    """
    if y_hat.ndim == 2 and y_hat.shape[1] == 1 and y.ndim == 1:
        return y.unsqueeze(1)
    if y_hat.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1)
    return y


# ------------------------- Runtime encoding --------------------------
@torch.no_grad()
def _maybe_runtime_encode(x4: torch.Tensor) -> torch.Tensor | None:
    """
    Si _RUNTIME_ENC['mode'] está activo y x es 4D (B,C,H,W), devuelve un tensor 5D (T,B,C,H,W)
    codificado en el device actual. Si no aplica, devuelve None.
    """
    enc = _RUNTIME_ENC
    mode, T, gain = enc.get("mode"), enc.get("T"), enc.get("gain")
    if mode is None or T is None or x4.ndim != 4:
        return None

    dev = x4.device

    # Normaliza a [0,1] con el orden correcto de argumentos en .to(...)
    if x4.dtype == torch.uint8:
        x = x4.to(device=dev, dtype=torch.float32) / 255.0
    else:
        x = x4.to(device=dev, dtype=torch.float32)
        # Si parece estar en 0..255 en float, escala; si no, sólo clamp
        x_max = float(torch.nan_to_num(x.detach().max(), nan=0.0).item()) if x.numel() > 0 else 1.0
        if x_max > 1.5:
            x = x / 255.0
        x = x.clamp_(0.0, 1.0)

    m = str(mode).lower()
    if m == "raw":
        # Repite la imagen 'T' veces: (B,C,H,W) -> (T,B,C,H,W)
        return x.unsqueeze(0).repeat(T, 1, 1, 1, 1).contiguous()

    if m == "rate":
        p = (x * float(gain if gain is not None else 1.0)).clamp_(0.0, 1.0)
        r = torch.rand((T, *x.shape), device=dev)
        spikes = (r < p.unsqueeze(0)).to(torch.float32)
        return spikes.contiguous()

    if m == "latency":
        # Un único spike cuya latencia decrece con la intensidad
        p = (x * float(gain if gain is not None else 1.0)).clamp_(0.0, 1.0)
        t_fire = torch.round((1.0 - p) * float(max(T - 1, 0))).to(torch.long)  # (B,C,H,W)
        N = t_fire.numel()
        spikes = torch.zeros((T, N), device=dev, dtype=torch.float32)
        ones = torch.ones((1, N), device=dev, dtype=torch.float32)
        spikes.scatter_(0, t_fire.view(1, -1), ones)  # coloca 1 en el instante t_fire por píxel
        spikes = spikes.view(T, *x.shape).contiguous()
        return spikes

    # Modo no reconocido -> no hacer nada
    return None


def _forward_with_cached_orientation(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    phase_hint: Dict[str, str],
    phase: str,
) -> torch.Tensor:
    """Estandariza entrada y hace un forward robusto:
    - Mueve x,y a device.
    - Si x es 4D y runtime encode está activo -> (T,B,C,H,W).
    - Los modelos esperan SIEMPRE (T,B,C,H,W).
    - Decide la orientación por shapes y cachea por fase.
    - Autocorrige una vez si la salida no tiene dim0 == B.
    """
    # mover a device
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    # 4D -> 5D (runtime) si procede
    if x.ndim == 4 and _RUNTIME_ENC.get("mode") is not None:
        x_rt = _maybe_runtime_encode(x)
        if x_rt is not None:
            x = x_rt  # (T,B,C,H,W)

    # Si no es secuencia, forward directo
    if x.ndim != 5:
        with autocast("cuda", enabled=use_amp):
            return model(x)

    B = int(y.shape[0])  # batch esperado en dim 1 (modelo espera (T,B,...))

    # Decisión por shapes: objetivo (T,B,...)
    hint = phase_hint.get(phase)
    if hint is None:
        if x.shape[1] == B:
            hint = "ok"        # ya está (T,B,...)
        elif x.shape[0] == B:
            hint = "permute"   # viene (B,T,...) -> permutar
        else:
            hint = "ok"        # por defecto
        phase_hint[phase] = hint

    x_fwd = x.permute(1, 0, 2, 3, 4).contiguous() if hint == "permute" else x

    with autocast("cuda", enabled=use_amp):
        y_hat = model(x_fwd)

    # Autocorrección si dim0 != B
    if isinstance(y_hat, torch.Tensor) and y_hat.ndim >= 1 and y_hat.shape[0] != B:
        if hint == "ok":
            # probar permutando
            x_alt = x.permute(1, 0, 2, 3, 4).contiguous()
            with torch.no_grad(), autocast("cuda", enabled=use_amp):
                y_try = model(x_alt)
            if isinstance(y_try, torch.Tensor) and y_try.ndim >= 1 and y_try.shape[0] == B:
                phase_hint[phase] = "permute"
                with autocast("cuda", enabled=use_amp):
                    y_hat = model(x_alt)
        else:
            # veníamos permutando; probar sin permutar
            with torch.no_grad(), autocast("cuda", enabled=use_amp):
                y_try = model(x)
            if isinstance(y_try, torch.Tensor) and y_try.ndim >= 1 and y_try.shape[0] == B:
                phase_hint[phase] = "ok"
                with autocast("cuda", enabled=use_amp):
                    y_hat = model(x)

    return y_hat



# ---------------------------------------------------------------------
# Configuración de entrenamiento
# ---------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 8
    lr: float = 1e-3
    amp: bool = True
    seed: Optional[int] = None  # reproducibilidad opcional
    # Early Stopping (opcional; si no usas, deja None/False)
    es_patience: Optional[int] = None
    es_min_delta: float = 0.0  # mejora mínima en val_loss para resetear paciencia


# ---------------------------------------------------------------------
# Entrenamiento supervisado de una tarea (con EWC opcional y EarlyStopping opcional)
# ---------------------------------------------------------------------
def train_supervised(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: nn.Module,
    cfg: TrainConfig,
    out_dir: Path,
    method=None,  # e.g., EWC con .penalty()
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rutas de checkpoints
    best_ckpt = out_dir / "best.pth"
    last_ckpt = out_dir / "last.pth"

    if cfg.seed is not None:
        set_seeds(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    use_amp = bool(cfg.amp and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp)

    # --- HISTORIA: añadimos val_mae y val_mse (mínimo impacto de rendimiento) ---
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_mse": []}
    t0 = time.time()

    best_val = float("inf")
    patience_left = cfg.es_patience if (cfg.es_patience is not None and cfg.es_patience > 0) else None
    best_state = None

    # Logging de it/s opcional vía env (no requiere parche externo)
    LOG_ITPS = os.environ.get("TRAIN_LOG_ITPS", "0") == "1"

    # Hints de orientación (solo se resuelven una vez por fase)
    phase_hint = {"train": None, "val": None}

    # --- contador global para logging periódico EWC
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        # it/s: cuenta iteraciones y tiempo de la fase de train
        it_count = 0
        t_epoch0 = time.perf_counter()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
            # IMPORTANT: mueve 'y' explícitamente al device aquí también, para evitar
            # que quede en CPU cuando se alinea contra y_hat:
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # Forward con detección de orientación cacheada + runtime encode si procede
            y_hat = _forward_with_cached_orientation(
                model=model,
                x=x,
                y=y,
                device=device,
                use_amp=use_amp,
                phase_hint=phase_hint,
                phase="train",
            )

            # Alinear target y asegurar device/dtype exactamente igual a y_hat
            y_aligned = _align_target_shape(y_hat, y)
            y_aligned = y_aligned.to(device=y_hat.device, dtype=y_hat.dtype, non_blocking=True)

            # Cálculo de pérdida base
            with autocast("cuda", enabled=use_amp):
                loss_base = loss_fn(y_hat, y_aligned)

            # Penalización (EWC u otras)
            pen = method.penalty() if method is not None else 0.0
            if isinstance(pen, torch.Tensor):
                pen = pen.to(device=y_hat.device, dtype=y_hat.dtype)
            else:
                pen = torch.tensor(float(pen), device=y_hat.device, dtype=y_hat.dtype)

            loss = loss_base + pen

            # --- Logging de diagnóstico (cada N steps) si hay método ---
            log_inner  = bool(getattr(method, "inner_verbose", False))
            log_every  = int(getattr(method, "inner_every", 50))
            if method is not None and log_inner and (global_step % max(1, log_every) == 0):
                base_val = float(loss_base.detach().item())
                pen_val  = float(pen.detach().item())
                ratio = pen_val / max(1e-8, base_val)
                lam_suggest_msg = ""
                try:
                    curr_lam = getattr(getattr(method, "impl", None), "cfg", None)
                    curr_lam = getattr(curr_lam, "lambd", None)
                    if curr_lam is not None and base_val > 0.0 and pen_val > 0.0:
                        target_ratio = 1.0
                        lam_next = float(curr_lam) * (target_ratio / max(1e-8, ratio))
                        lam_suggest_msg = (
                            f" | λ_actual={float(curr_lam):.3e} → λ_sugerido≈{lam_next:.3e} (target pen/base={target_ratio})"
                        )
                except Exception:
                    pass
                print(f"[EWC] base={base_val:.4g} | pen={pen_val:.4g} | pen/base={ratio:.3f}{lam_suggest_msg}")

            # Backward + step (AMP seguro)
            if use_amp:
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
            global_step += 1
            it_count += 1

        # it/s (sincroniza GPU para medir bien)
        if LOG_ITPS and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_epoch0
        if LOG_ITPS and dt > 0:
            ips = it_count / dt
            print(f"[TRAIN it/s] epoch {epoch}/{cfg.epochs}: {ips:.1f} it/s  ({it_count} iters en {dt:.2f}s)")

        train_loss = running / max(1, len(train_loader))

        # ------------------- Validación (MSE + MAE en una sola pasada) -------------------
        model.eval()
        v_running_mse = 0.0
        v_running_mae = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                # Mueve y a device
                y = y.to(device, non_blocking=True)

                y_hat = _forward_with_cached_orientation(
                    model=model,
                    x=x,
                    y=y,
                    device=device,
                    use_amp=use_amp,
                    phase_hint=phase_hint,
                    phase="val",
                )
                y_aligned = _align_target_shape(y_hat, y).to(device=y_hat.device, dtype=y_hat.dtype, non_blocking=True)
                # MSE por batch (como antes)
                with autocast("cuda", enabled=use_amp):
                    v_loss = loss_fn(y_hat, y_aligned)
                v_running_mse += float(v_loss.detach().item())
                # MAE por batch (coste ínfimo; en FP32 para estabilidad)
                mae_batch = torch.abs(y_hat.to(torch.float32) - y_aligned.to(torch.float32)).mean()
                v_running_mae += float(mae_batch.detach().item())
                n_val_batches += 1

        val_loss = v_running_mse / max(1, n_val_batches)  # MSE medio por batch
        val_mae  = v_running_mae / max(1, n_val_batches)  # MAE medio por batch

        # --- guardar historia por epoch ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_mse"].append(val_loss)  # coherente: el val_loss ya es MSE medio

        # guarda "last" cada epoch
        torch.save(model.state_dict(), last_ckpt)

        # Early stopping (y "best") usa val_loss (MSE)
        if patience_left is not None:
            improved = (best_val - val_loss) > cfg.es_min_delta
            if improved:
                best_val = val_loss
                patience_left = cfg.es_patience
                # guarda best en CPU para ahorrar VRAM
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_ckpt)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    elapsed = time.time() - t0

    # Si no hubo mejora nunca, al menos guarda best = last
    if not best_ckpt.exists():
        torch.save(model.state_dict(), best_ckpt)

    # Manifest con metadatos (incluye history con val_mae/val_mse)
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
        "checkpoints": {
            "best": str(best_ckpt),
            "last": str(last_ckpt),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return history
