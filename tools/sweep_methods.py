# tools/sweep_methods.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import gc
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List

# ── NUEVO: señal y volcados
import os
import signal
import faulthandler
faulthandler.enable()  # habilita dump de trazas bajo demanda

# Asegura que 'src' esté en el sys.path ejecutando desde la raíz del repo
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_preset
from src.utils_components import build_components_for
from src.runner import run_continual

# ------------------------------- carga del sweep -------------------------------
_POSSIBLE_KEYS = ("runs", "exps", "experiments", "sweep")

def _read_sweep(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe el sweep file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) Si ya es lista -> OK
    if isinstance(data, list):
        return data
    # 2) Si es dict, prueba claves comunes
    if isinstance(data, dict):
        for k in _POSSIBLE_KEYS:
            v = data.get(k, None)
            if isinstance(v, list):
                return v
        # 3) Si hay exactamente una clave cuyo valor es lista -> úsala
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if len(list_keys) == 1:
            return data[list_keys[0]]
        # 4) Si hay varias listas, elige la primera que parezca lista de runs
        for k in list_keys:
            v = data[k]
            if all(isinstance(x, (str, dict)) for x in v):
                return v
        # 5) No se pudo inferir
        keys_preview = ", ".join(list(data.keys())[:8])
        raise TypeError(
            "El sweep debe ser una lista de runs o un dict con una lista en "
            f"alguna de estas claves: {', '.join(_POSSIBLE_KEYS)}. "
            f"Claves detectadas: {keys_preview}"
        )
    raise TypeError("El sweep debe ser una lista o un dict con una lista de runs.")

def _norm_run(run_spec):
    """Devuelve dict canónico con keys:
      method(str), params(dict), tag(str|None),
      preset(str|None), seed(int|None), amp(bool|None)
    """
    if isinstance(run_spec, str):
        return {
            "method": run_spec,
            "params": {},
            "tag": None,
            "preset": None,
            "seed": None,
            "amp": None,
        }
    if isinstance(run_spec, dict):
        # Nombre del método
        method = run_spec.get("method", run_spec.get("name", None))
        if not isinstance(method, str) or not method:
            raise ValueError(f"Run mal formado: falta 'method' en {run_spec}")

        # Preferimos 'params', pero aceptamos 'method_kwargs' como alias común
        params = run_spec.get("params", None)
        if params is None:
            params = run_spec.get("method_kwargs", {}) or {}
        if not isinstance(params, dict):
            raise ValueError(f"'params'/'method_kwargs' debe ser dict en {run_spec}")

        # Tag opcional
        tag = run_spec.get("tag", None)
        tag = str(tag) if tag is not None else None

        # Overrides opcionales
        preset = run_spec.get("preset", None)
        preset = str(preset) if preset is not None else None
        seed = run_spec.get("seed", None)
        amp  = run_spec.get("amp",  None)

        return {
            "method": method,
            "params": params,
            "tag": tag,
            "preset": preset,
            "seed": seed,
            "amp": amp,
        }
    raise TypeError(f"Tipo de run no soportado: {type(run_spec)}")

def _build_task_list_from_cfg(cfg):
    prep = cfg.get("prep", {}) or {}
    runs = prep.get("runs", None)
    if not runs:
        runs = ["circuito1", "circuito2"]
    return [{"name": str(r)} for r in runs]

# ------------------------ ejecución aislada por subproceso ------------------------
def _child_run(run_norm: Dict[str, Any], eff_preset: str, out_root: str, safe_dl: int, conn):
    """
    Ejecuta un run en un subproceso y devuelve por pipe:
      {"ok": True, "run_dir": str} o {"ok": False, "error": str, "traceback": str}
    Con handlers de señal para volcado de memoria/trazas si el SO envía SIGTERM/SIGINT.
    """
    # — NUEVO: handlers que vuelcan información si el proceso es terminado —
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    def _dump_termination(signum: int, frame):
        try:
            pid = os.getpid()
            logf = out_root_path / f"_terminated_{pid}.log"
            with logf.open("w", encoding="utf-8") as f:
                f.write(f"[SIGNAL] {signum}\n")
                # CPU memoria
                try:
                    import resource
                    ru = resource.getrusage(resource.RUSAGE_SELF)
                    f.write(f"ru_maxrss_kb={ru.ru_maxrss}\n")
                except Exception:
                    pass
                try:
                    import psutil
                    p = psutil.Process(pid)
                    f.write(f"rss_mb={p.memory_info().rss/1024**2:.2f}\n")
                except Exception:
                    pass
                # CUDA memoria
                try:
                    import torch
                    if torch.cuda.is_available():
                        f.write("\n[torch.cuda.memory_summary]\n")
                        f.write(torch.cuda.memory_summary(device=None, abbreviated=False))
                except Exception:
                    pass
                f.write("\n[faulthandler]\n")
                try:
                    faulthandler.dump_traceback(file=f, all_threads=True)
                except Exception:
                    pass
            # salida limpia con código tipo "128+signal"
        finally:
            os._exit(128 + signum)

    # registrar handlers
    try:
        signal.signal(signal.SIGTERM, _dump_termination)
        signal.signal(signal.SIGINT,  _dump_termination)
    except Exception:
        pass

    # ───────────────────────────────────────────────────────────────────────
    try:
        import torch

        cfg = load_preset(eff_preset)

        # Continual
        cfg.setdefault("continual", {})
        cfg["continual"]["method"] = run_norm["method"]
        merged = dict(cfg["continual"].get("params", {}) or {})
        merged.update(run_norm["params"] or {})
        cfg["continual"]["params"] = merged

        # Tag
        if run_norm["tag"]:
            cfg.setdefault("naming", {})
            cfg["naming"]["tag"] = run_norm["tag"]

        # Overrides seed/amp
        if run_norm["seed"] is not None:
            cfg.setdefault("data", {})
            cfg["data"]["seed"] = int(run_norm["seed"])
        if run_norm["amp"] is not None:
            cfg.setdefault("optim", {})
            cfg["optim"]["amp"] = bool(run_norm["amp"])

        # Safe dataloader niveles (opcional)
        # 0: no tocar; 1: persistent_workers=False; 2: además workers=0 y pin_memory=False
        if safe_dl >= 1:
            cfg.setdefault("data", {})
            cfg["data"]["persistent_workers"] = False
        if safe_dl >= 2:
            cfg["data"]["num_workers"] = 0
            cfg["data"]["pin_memory"] = False

        # Componentes y tasks
        make_loader_fn, make_model_fn, tfm = build_components_for(cfg)
        task_list = _build_task_list_from_cfg(cfg)

        # Ejecuta
        out_dir, _ = run_continual(
            task_list=task_list,
            make_loader_fn=make_loader_fn,
            make_model_fn=make_model_fn,
            tfm=tfm,
            cfg=cfg,
            preset_name=(cfg.get('_meta', {}).get('preset_key') or eff_preset),
            out_root=out_root,
            verbose=True,
        )

        # Limpieza explícita al finalizar el hijo
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

        conn.send({"ok": True, "run_dir": str(out_dir)})

    except Exception as e:
        tb = traceback.format_exc()
        conn.send({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": tb})
    finally:
        try:
            conn.close()
        except Exception:
            pass

# ---------------------------------- CLI main ----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, help="Nombre ('fast'/'std'/'accurate') o ruta a YAML.")
    ap.add_argument("--sweep-file", required=True, help="Ruta a JSON con runs.")
    ap.add_argument("--out", default="outputs", help="Carpeta de salida raíz.")
    ap.add_argument("--isolate", default="on", choices=["on", "off"],
                    help="Aísla cada run en un subproceso (recomendado para evitar fugas de shm/tmp).")
    ap.add_argument("--safe-dataloader", type=int, default=0, choices=[0, 1, 2],
                    help="Mitigaciones de DataLoader: 0=no tocar; 1=persistent_workers=False; 2=además workers=0 y pin_memory=False.")
    ap.add_argument("--sleep", type=float, default=0.5, help="Pausa entre runs (seg).")
    args = ap.parse_args()

    # 1) Lee y normaliza runs
    runs_in = _read_sweep(Path(args.sweep_file))
    runs = [_norm_run(r) for r in runs_in]

    out_root = Path(args.out)

    # PRE-LISTADO indicando qué preset se usará finalmente por run
    print(f"[INFO] {len(runs)} runs a ejecutar. OUT={out_root}")
    for i, r in enumerate(runs, start=1):
        pretty = f"{r['method']} :: {r['tag']}" if r["tag"] else r["method"]
        eff_preset = r["preset"] or args.preset
        print(f"  {i:02d}. {pretty}  (preset={eff_preset})")
    print()

    # 2) Ejecuta
    for i, r in enumerate(runs, start=1):
        eff_preset = r["preset"] or args.preset
        pretty = f"{r['method']} :: {r['tag']}" if r["tag"] else r["method"]
        print(f"\n=== [{i}/{len(runs)}] {r['method']} | tag={r['tag'] or 'exps'} | preset={eff_preset} ===\n")

        if args.isolate == "off":
            # Modo “inline”
            try:
                cfg = load_preset(eff_preset)
                cfg.setdefault("continual", {})
                cfg["continual"]["method"] = r["method"]
                merged = dict(cfg["continual"].get("params", {}) or {})
                merged.update(r["params"] or {})
                cfg["continual"]["params"] = merged
                if r["tag"]:
                    cfg.setdefault("naming", {})
                    cfg["naming"]["tag"] = r["tag"]
                if r["seed"] is not None:
                    cfg.setdefault("data", {})
                    cfg["data"]["seed"] = int(r["seed"])
                if r["amp"] is not None:
                    cfg.setdefault("optim", {})
                    cfg["optim"]["amp"] = bool(r["amp"])

                make_loader_fn, make_model_fn, tfm = build_components_for(cfg)
                task_list = _build_task_list_from_cfg(cfg)

                run_continual(
                    task_list=task_list,
                    make_loader_fn=make_loader_fn,
                    make_model_fn=make_model_fn,
                    tfm=tfm,
                    cfg=cfg,
                    preset_name=(cfg.get('_meta', {}).get('preset_key') or eff_preset),
                    out_root=out_root,
                    verbose=True,
                )
            except Exception as e:
                print(f"[ERROR] {type(e).__name__}: {e}")
            # Limpieza “fuerte” entre runs
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()
        else:
            # Modo aislado (recomendado)
            p_conn, c_conn = mp.Pipe(duplex=False)
            proc = mp.Process(
                target=_child_run,
                args=(r, eff_preset, str(out_root), int(args.safe_dataloader), c_conn),
                daemon=False,
            )
            proc.start()
            c_conn.close()
            try:
                msg = p_conn.recv()
                if isinstance(msg, dict) and msg.get("ok"):
                    print("[OK] run_dir:", msg.get("run_dir"))
                else:
                    err = (msg or {}).get("error", "Unknown error")
                    print(f"[ERROR] {err}")
                    tb = (msg or {}).get("traceback", "")
                    if tb:
                        print(tb)
            except EOFError:
                print("[ERROR] Proceso hijo terminó sin enviar resultado.")
            finally:
                proc.join()
                p_conn.close()

        time.sleep(float(args.sleep))
    return 0

if __name__ == "__main__":
    sys.exit(main())
