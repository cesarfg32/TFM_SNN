# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy

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
    """
    Devuelve dict canónico con keys:
      method(str), params(dict), tag(str|None),
      preset(str|None), seed(int|None), amp(bool|None)

    Admite:
      - "ewc"
      - {"method":"ewc", "params": {...}, "tag": "..."}
      - {"method":"ewc", "method_kwargs": {...}, "tag": "..."}  # compat
      - y overrides opcionales: "preset", "seed", "amp"
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

# ---------------------------------- CLI main ----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, help="Nombre ('fast'/'std'/'accurate') o ruta a YAML.")
    ap.add_argument("--sweep-file", required=True, help="Ruta a JSON con runs.")
    ap.add_argument("--out", default="outputs", help="Carpeta de salida raíz.")
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
        # Carga preset efectivo del run (o el del CLI si no especifica)
        eff_preset = r["preset"] or args.preset
        cfg = load_preset(eff_preset)

        # Overrides de continual
        cfg.setdefault("continual", {})
        cfg["continual"]["method"] = r["method"]
        merged = dict(cfg["continual"].get("params", {}) or {})
        merged.update(r["params"] or {})
        cfg["continual"]["params"] = merged

        # Tag
        if r["tag"]:
            cfg.setdefault("naming", {})
            cfg["naming"]["tag"] = r["tag"]

        # Overrides de seed/amp si vienen en el run
        if r["seed"] is not None:
            cfg.setdefault("data", {})
            cfg["data"]["seed"] = int(r["seed"])
        if r["amp"] is not None:
            cfg.setdefault("optim", {})
            cfg["optim"]["amp"] = bool(r["amp"])

        # Como el preset puede cambiar por run, construimos componentes por run
        make_loader_fn, make_model_fn, tfm = build_components_for(cfg)
        task_list = _build_task_list_from_cfg(cfg)

        pretty = f"{r['method']} :: {r['tag']}" if r["tag"] else r["method"]
        try:
            print(f"\n=== [{i}/{len(runs)}] {r['method']} | tag={r['tag'] or 'exps'} | preset={eff_preset} ===\n")
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

    return 0

if __name__ == "__main__":
    sys.exit(main())
