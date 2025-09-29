# tools/run_continual.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_preset, build_task_list_for, build_components_for
from src.runner import run_continual

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/presets.yaml"))
    ap.add_argument("--preset", required=True, choices=["fast", "std", "accurate"])
    ap.add_argument("--tasks-file", type=Path, default=None,
                    help="Override: ruta a tasks.json / tasks_balanced.json")
    ap.add_argument("--tag", default="", help="Etiqueta extra para el nombre de salida")
    args = ap.parse_args()

    cfg = load_preset(args.config, args.preset)
    if args.tag:
        cfg.setdefault("naming", {})["tag"] = args.tag

    # Componentes coherentes con el preset
    tfm, make_loader_fn, make_model_fn = build_components_for(cfg, ROOT)

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
