#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep de métodos (versión CL real):
- Un método = UN solo proceso que recorre TODAS las tareas (circuito1 -> circuito2).
- --safe-dataloader opcional para mitigar problemas de workers/WSL.
- Logs sin buffering (-u y PYTHONUNBUFFERED=1) propagados al subproceso.
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List

# ===============================
# 0) Asegura que 'src' está en sys.path
# ===============================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===============================
# 1) Bootstrap multiproceso CUDA
# ===============================
def _ensure_spawn_for_cuda() -> None:
    """Garantiza 'spawn' como start method ANTES de que se importe torch/cuda."""
    try:
        cur = mp.get_start_method(allow_none=True)
    except Exception:
        cur = None
    if cur != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            if cur not in (None, "spawn"):
                print(f"[WARN] start_method era '{cur}'. Con CUDA se recomienda 'spawn'.", file=sys.stderr)

_ensure_spawn_for_cuda()

# Opcionales útiles (no obligatorios):
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ===============================
# 2) CLI
# ===============================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Barrido de métodos SNN/CL (launcher CL real)")
    p.add_argument("--preset", type=str, required=True, help="Nombre del preset (p.ej. fast | std | accurate)")
    p.add_argument("--sweep-file", type=str, required=True, help="Ruta a JSON de experimentos")
    p.add_argument("--outdir", type=str, default="outputs", help="Directorio de resultados")
    p.add_argument("--safe-dataloader", type=int, choices=[0,1], default=0,
                   help="Si 1, aplica overrides conservadores a DataLoader (workers=0, sin persistent_workers).")
    p.add_argument("--no-amp", action="store_true", help="Desactiva AMP (triage rápido).")
    p.add_argument("--dry-run", action="store_true", help="No ejecuta; imprime comandos.")
    return p

# ===============================
# 3) Utilidades de UX/Logs
# ===============================
def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _print_header(nruns: int, outdir: Path):
    print(f"[INFO] {nruns} runs a ejecutar. OUT={outdir}\n")

def _print_method(idx: int, total: int, method_name: str, tag: str, preset: str):
    print(f"  {idx:02d}. {method_name} :: {tag}  (preset={preset})")
    print(f"\n=== [{idx}/{total}] {method_name} | tag={tag} | preset={preset} ===\n")

# ===============================
# 4) Carga/expansión del sweep
# ===============================
def load_sweep(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "experiments" in data:
        exps = data["experiments"]
    else:
        exps = data
    if not isinstance(exps, list):
        raise ValueError(f"Formato de sweep inválido en {path}")
    return exps

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _expand_by_seeds(exps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Permite usar: { "seeds":[42,777], "tag":"...", ... } en cada experimento."""
    out: List[Dict[str, Any]] = []
    for e in exps:
        seeds = e.get("seeds", None)
        if not seeds:
            out.append(e)
            continue
        base_tag = e.get("tag", "")
        for s in seeds:
            e2 = json.loads(json.dumps(e))  # copia profunda
            e2.pop("seeds", None)
            # override de seed en data
            ov = e2.get("overrides", {})
            ov = _deep_merge(ov, {"data": {"seed": int(s)}})
            e2["overrides"] = ov
            # tag con sufijo de seed
            if base_tag:
                e2["tag"] = f"{base_tag}_s{s}"
            else:
                e2["tag"] = f"s{s}"
            out.append(e2)
    return out

# ===============================
# 5) Overrides de DataLoader
# ===============================
def dataloader_overrides(safe_flag: int) -> Dict[str, Any]:
    if not safe_flag:
        return {}
    return {
        "data": {
            "num_workers": 0,
            "prefetch_factor": None,
            "pin_memory": False,
            "persistent_workers": False,
        }
    }

# ===============================
# 6) Ejecución de un run (UN solo proceso recorre todas las tareas)
# ===============================
def run_one(
    exp: Dict[str, Any],
    preset: str,
    outdir: Path,
    safe_loader: int,
    no_amp: bool,
    dry: bool
) -> int:
    method_name: str = exp.get("method", "naive")
    tag: str = exp.get("tag", f"{preset}_{method_name}")

    # params (mantenemos compat anterior sin cambiar nombres en el sweep)
    params: Dict[str, Any] = (exp.get("params") or exp.get("method_kwargs") or {})

    ov = {
        "preset": preset,
        "naming": {"tag": tag},
        "method": {"name": method_name, **params},
        "tasks": ["circuito1", "circuito2"],
    }

    # safe dataloader (desde flag del CLI)
    ov = _deep_merge(ov, dataloader_overrides(safe_loader))

    # Overrides arbitrarios por experimento (NUEVO)
    if "overrides" in exp:
        ov = _deep_merge(ov, exp["overrides"])

    # AMP override por experimento y por flag global
    if "amp" in exp:
        ov = _deep_merge(ov, {"optim": {"amp": bool(exp["amp"])}})
    if no_amp:
        ov = _deep_merge(ov, {"optim": {"amp": False}})

    cfg_path = outdir / f"tmp_cfg_{tag}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(ov, f, ensure_ascii=False, indent=2)

    # Lanza el RUNNER (entrada única)
    cmd = [
        sys.executable, "-u",
        "-m", "src.runner",
        "--config", str(cfg_path),
    ]
    if dry:
        print("[DRY] CMD:", " ".join(cmd))
        return 0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TQDM_MININTERVAL", "0.1")

    ret = subprocess.call(cmd, env=env)
    return ret

# ===============================
# 7) Main
# ===============================
def main() -> int:
    args = build_argparser().parse_args()
    sweep_path = Path(args.sweep_file)
    outdir = Path(args.outdir)

    exps = load_sweep(sweep_path)
    exps = _expand_by_seeds(exps)   # <<--- NUEVO

    _print_header(len(exps), outdir)

    for i, exp in enumerate(exps, 1):
        method = exp.get("method", "naive")
        tag = exp.get("tag", f"{args.preset}_{method}")
        _print_method(i, len(exps), method, tag, args.preset)

        ret = run_one(
            exp=exp,
            preset=args.preset,
            outdir=outdir,
            safe_loader=args.safe_dataloader,
            no_amp=args.no_amp,
            dry=args.dry_run,
        )
        if ret != 0:
            print(f"[{_stamp()}] ERROR: run {i}/{len(exps)} falló con código {ret}", file=sys.stderr)
            return ret

    print(f"[{_stamp()}] OK: todos los runs finalizados.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
