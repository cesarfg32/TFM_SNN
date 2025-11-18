# tools/run_continual.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_preset
from src.runner import run_continual
from src.utils import build_task_list_for
from src.utils_components import build_components_for

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/presets.yaml"))
    ap.add_argument("--preset", required=True, choices=["fast", "std", "accurate"])
    ap.add_argument("--tasks-file", type=Path, default=None,
                    help="Override: ruta a tasks.json / tasks_balanced.json")
    ap.add_argument("--tag", default="", help="Etiqueta extra para el nombre de salida")
    ap.add_argument("--method", default=None,
                    help="Override: nombre del método continual (p.ej. 'as-snn', 'ewc', 'rehearsal+ewc').")
    ap.add_argument("--params", default="",
                    help="Override: key=val[,key=val...] para method params. Ej: 'gamma_ratio=0.3,lambda_a=1.6,ema=0.82'")
    args = ap.parse_args()

    cfg = load_preset(args.config, args.preset)
    if args.tag:
        cfg.setdefault("naming", {})["tag"] = args.tag
    # Overrides de método/params
    if args.method:
        cfg.setdefault("continual", {})["method"] = args.method
    if args.params:
        # parseo simple key=val[,key=val...], convierte a float/int si procede
        out = {}
        for kv in args.params.split(","):
            if not kv.strip():
                continue
            k, v = kv.split("=", 1)
            v = v.strip()
            # castear números
            try:
                if "." in v or "e" in v.lower():
                    v_cast = float(v)
                else:
                    v_cast = int(v)
            except ValueError:
                # bools simples
                if v.lower() in ("true","false"):
                    v_cast = (v.lower() == "true")
                else:
                    v_cast = v
            out[k.strip()] = v_cast
        cfg.setdefault("continual", {})["params"] = out

    # Componentes coherentes con el preset
    make_loader_fn, make_model_fn, tfm = build_components_for(cfg)

    # Task list (override si pasas --tasks-file)
    if args.tasks_file and args.tasks_file.exists():
        tasks_file = args.tasks_file
        tasks_json = json.loads(tasks_file.read_text(encoding="utf-8"))
        task_list = [{"name": n, "paths": tasks_json["splits"][n]} for n in tasks_json["tasks_order"]]
    else:
        task_list, tasks_file = build_task_list_for(cfg, ROOT)

    out_dir, _ = run_continual(
        task_list=task_list,
        make_loader_fn=make_loader_fn,
        make_model_fn=make_model_fn,
        tfm=tfm,
        cfg=cfg,
        preset_name=args.preset,
        out_root=ROOT / "outputs",
        verbose=True,
    )
    print("OK:", out_dir)

if __name__ == "__main__":
    main()
