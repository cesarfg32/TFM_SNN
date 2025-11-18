from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

# Asegura import del paquete
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner import run_from_config_dict

def _parse_params(kv_text: str) -> dict:
    if not kv_text:
        return {}
    out = {}
    for kv in kv_text.split(","):
        if not kv.strip():
            continue
        k, v = kv.split("=", 1)
        v = v.strip()
        try:
            if "." in v or "e" in v.lower():
                v_cast = float(v)
            else:
                v_cast = int(v)
        except ValueError:
            if v.lower() in ("true","false"):
                v_cast = (v.lower() == "true")
            else:
                v_cast = v
        out[k.strip()] = v_cast
    return out

def main():
    ap = argparse.ArgumentParser(description="Thin wrapper → src.runner.run_from_config_dict")
    ap.add_argument("--config", type=Path, default=Path("configs/presets.yaml"),
                    help="Ruta al YAML de presets (opcional; por defecto el estándar)")
    ap.add_argument("--preset", required=True, choices=["fast", "std", "accurate"])
    ap.add_argument("--tasks-file", type=Path, default=None,
                    help="Override: ruta a tasks.json / tasks_balanced.json (se usará su order para 'tasks')")
    ap.add_argument("--tag", default="", help="Etiqueta extra para el nombre de salida")
    ap.add_argument("--method", default=None,
                    help="Override: nombre del método continual (p.ej. 'as-snn', 'ewc', 'rehearsal+ewc').")
    ap.add_argument("--params", default="",
                    help="Override: key=val[,key=val...] para method params. Ej: 'gamma_ratio=0.3,lambda_a=1.6,ema=0.82'")
    args = ap.parse_args()

    cfg_in = {
        "config_path": str(args.config) if args.config else None,
        "preset": args.preset,
    }

    if args.tag:
        cfg_in["naming"] = {"tag": args.tag}

    if args.method or args.params:
        cfg_in["method"] = {"name": args.method or "naive", **_parse_params(args.params)}

    # Si se pasa un tasks-file, extraemos los nombres en orden y los pasamos como 'tasks'
    if args.tasks_file and args.tasks_file.exists():
        tasks_json = json.loads(args.tasks_file.read_text(encoding="utf-8"))
        tasks_order = list(tasks_json.get("tasks_order", []))
        # A runner le basta con los nombres; completará paths desde tasks.json canónico
        cfg_in["tasks"] = tasks_order

    out_dir = run_from_config_dict(cfg_in)
    print("OK:", out_dir)

if __name__ == "__main__":
    main()
