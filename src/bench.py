# src/bench.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import time, torch
from contextlib import nullcontext
import src.training as training

def to_5d(xb, encoder: str, T: int, gain: float, device):
    """Devuelve (x5d, used_runtime) desde 4D o 5D."""
    if xb.ndim == 5:
        return xb.permute(1,0,2,3,4).contiguous(), False
    elif xb.ndim == 4:
        training.set_encode_runtime(mode=encoder, T=T, gain=gain, device=device)
        x5d = training._permute_if_needed(xb)  # encode+permute
        return x5d, True
    else:
        raise RuntimeError(f"Batch shape inesperada: {xb.shape}")

def forward_once_ms(model, x5d, device, use_amp=True):
    """Mide un forward único (ms)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.amp.autocast('cuda', enabled=(use_amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), ctx:
        _ = model(x5d.to(device, non_blocking=True))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0

def loop_gpu_only_its(model, x5d, device, iters=100, use_amp=True):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.amp.autocast('cuda', enabled=(use_amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), ctx:
        x5d_dev = x5d.to(device, non_blocking=True)
        for _ in range(iters):
            _ = model(x5d_dev)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return iters / (time.perf_counter() - t0)

def pipeline_its(model, loader, device, iters=100, use_amp=True, encoder=None, T=None, gain=None):
    """Itera loader+modelo (activa runtime encode si el loader es 4D)."""
    it = iter(loader)
    try:
        xb0, _ = next(it)
    except StopIteration:
        return float('nan')

    used_rt = False
    if xb0.ndim == 4:
        training.set_encode_runtime(mode=encoder, T=T, gain=gain, device=device)
        used_rt = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.amp.autocast('cuda', enabled=(use_amp and torch.cuda.is_available())) if torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), ctx:
        # primero
        x = xb0.permute(1,0,2,3,4).contiguous() if xb0.ndim==5 else training._permute_if_needed(xb0)
        _ = model(x.to(device, non_blocking=True))
        done = 1
        # resto
        while done < iters:
            try:
                xb, _ = next(it)
            except StopIteration:
                it = iter(loader)
                xb, _ = next(it)
            x = xb.permute(1,0,2,3,4).contiguous() if xb.ndim==5 else training._permute_if_needed(xb)
            _ = model(x.to(device, non_blocking=True))
            done += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    its = iters / (time.perf_counter() - t0)

    if used_rt:
        training.set_encode_runtime(None)
    return its

from pathlib import Path
from contextlib import nullcontext
import time
import json, torch
from torch import nn, optim
from torch.amp import autocast, GradScaler

import src.training as training
from src.utils import build_make_loader_fn, set_seeds

# ---------- Loader factory para notebooks ----------
def make_loader_fn_factory(
    ROOT: Path,
    **defaults  # ej. RUNTIME_ENCODE, SEED, num_workers, prefetch_factor, pin_memory, persistent_workers, aug_train, use_online_balancing, ...
):
    """
    Devuelve una función make_loader_fn(...) que usa el builder unificado de utils
    y mezcla correctamente defaults del notebook con overrides de cada llamada,
    sin duplicar num_workers/pin_memory/etc.
    """
    ROOT = Path(ROOT)

    # Flags de codificación (soporta ambos nombres por compatibilidad del propio notebook)
    use_offline_spikes = bool(defaults.pop("USE_OFFLINE_SPIKES", False))
    encode_runtime     = bool(defaults.pop("RUNTIME_ENCODE", not use_offline_spikes))

    # Mapea nombre antiguo a nuevo (si lo usan en el notebook)
    if "use_online_balancing" in defaults:
        uob = bool(defaults.pop("use_online_balancing"))
        if uob:
            defaults.setdefault("balance_train", True)

    base_mk = build_make_loader_fn(
        root=ROOT,
        use_offline_spikes=use_offline_spikes,
        encode_runtime=encode_runtime,
    )

    def make_loader_fn(task, batch_size, encoder, T, gain, tfm, seed, **overrides):
        dl_kwargs = {**defaults, **overrides}  # utils filtrará lo que no toque
        return base_mk(
            task=task, batch_size=batch_size, encoder=encoder, T=T, gain=gain, tfm=tfm, seed=seed,
            **dl_kwargs
        )

    return make_loader_fn

# ---------- Prueba universal de forward con runtime encode si aplica ----------
def universal_smoke_forward(
    make_loader_fn,
    task: dict,
    *,
    encoder: str,
    T: int,
    gain: float,
    tfm,
    seed: int,
    device: torch.device,
    use_encode_runtime: bool,
) -> tuple[torch.Size, torch.Size]:
    """Devuelve (x5d_shape, yhat_shape). Imprime qué camino se ha tomado."""
    tr, _, _ = make_loader_fn(
        task=task, batch_size=8, encoder=encoder, T=T, gain=gain, tfm=tfm, seed=seed
    )
    xb, yb = next(iter(tr))
    used_runtime = False

    if xb.ndim == 5:
        x5d = xb.permute(1, 0, 2, 3, 4).contiguous()
        print("dataset ya codificado; solo permuto a (T,B,C,H,W)")
    elif xb.ndim == 4:
        if not use_encode_runtime:
            raise RuntimeError("El loader entrega 4D pero use_encode_runtime=False. Actívalo o usa encoder temporal en el dataset.")
        training.set_encode_runtime(mode=encoder, T=T, gain=gain, device=device)
        x5d = training._permute_if_needed(xb)  # encode + permute
        used_runtime = True
        print("dataset 4D; uso encode en GPU y permuto a (T,B,C,H,W)")
    else:
        raise RuntimeError(f"Forma inesperada del batch: {tuple(xb.shape)}")

    print("x5d.device:", x5d.device, "| shape:", tuple(x5d.shape))

    # Modelo pequeño para prueba rápida
    from src.models import MiniSNN
    model = MiniSNN(in_channels=(1 if tfm.to_gray else 3), lif_beta=0.95).to(device).eval()

    # AMP solo si hay CUDA
    ctx = autocast('cuda', enabled=torch.cuda.is_available()) if torch.cuda.is_available() else nullcontext()
    try:
        with torch.inference_mode(), ctx:
            y = model(x5d.to(device, non_blocking=True))
        print("[forward] ejecutado con AMP" if torch.cuda.is_available() else "[forward] ejecutado en FP32")
    finally:
        if used_runtime:
            training.set_encode_runtime(None)

    return x5d.shape, y.shape

# ---------- Toggle para imprimir it/s por época ----------
_ORIG_TRAIN = None

def enable_epoch_ips():
    """Reemplaza training.train_supervised por una versión que imprime it/s por época."""
    global _ORIG_TRAIN
    if _ORIG_TRAIN is not None:
        return  # ya activo
    _ORIG_TRAIN = training.train_supervised

    def train_supervised_ips(model: nn.Module, train_loader, val_loader, loss_fn: nn.Module, cfg, out_dir: Path, method=None):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.seed is not None:
            set_seeds(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=cfg.lr)

        use_amp = bool(cfg.amp and torch.cuda.is_available())
        scaler = GradScaler(enabled=use_amp)

        history = {"train_loss": [], "val_loss": []}
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running = 0.0
            nb = 0
            t0 = time.perf_counter()

            for x, y in train_loader:
                x = training._permute_if_needed(x).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=use_amp):
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y)
                    if method is not None:
                        loss = loss + method.penalty()

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                running += loss.item()
                nb += 1

            epoch_time = time.perf_counter() - t0
            ips = nb / epoch_time if epoch_time > 0 else float("nan")
            print(f"[TRAIN it/s] epoch {epoch}/{cfg.epochs}: {ips:.1f} it/s ({nb} iters en {epoch_time:.2f}s)")

            # validación
            model.eval()
            v_running = 0.0; nvb = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = training._permute_if_needed(x).to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with autocast("cuda", enabled=use_amp):
                        y_hat = model(x)
                        v_loss = loss_fn(y_hat, y)
                    v_running += v_loss.item(); nvb += 1

            history["train_loss"].append(running / max(1, nb))
            history["val_loss"].append(v_running / max(1, nvb))

        # persistir manifest mínimo (sin repetir lo de training original)
        (Path(out_dir) / "manifest.json").write_text(
            json.dumps({
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "amp": cfg.amp,
                "seed": cfg.seed,
                "device": str(device),
                "history": history,
            }, indent=2), encoding="utf-8"
        )
        return history

    training.train_supervised = train_supervised_ips

def disable_epoch_ips():
    """Restaura training.train_supervised original."""
    global _ORIG_TRAIN
    if _ORIG_TRAIN is None:
        return
    training.train_supervised = _ORIG_TRAIN
    _ORIG_TRAIN = None

def print_bench_config(*, NUM_WORKERS, PREFETCH, PIN_MEMORY, PERSISTENT, USE_ONLINE_BALANCING):
    print(
        f"[Bench workers={NUM_WORKERS} prefetch={PREFETCH} "
        f"pin={PIN_MEMORY} persistent={PERSISTENT} | "
        f"online_bal={USE_ONLINE_BALANCING}"
    )
