# tools/run_continual.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_preset, build_make_loader_fn
from src.models import build_model
from src.datasets import ImageTransform
from src.runner import run_continual

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/presets.yaml"))
    ap.add_argument("--preset", required=True, choices=["fast","std","accurate"])
    ap.add_argument("--tasks-file", type=Path, default=Path("data/processed/tasks.json"))
    ap.add_argument("--tag", default="", help="Etiqueta extra para el nombre de salida")
    args = ap.parse_args()

    cfg = load_preset(args.config, args.preset)
    if args.tag:
        cfg.setdefault("naming", {})["tag"] = args.tag

    # tfm desde preset
    mw, mh, to_gray = cfg["model"]["img_w"], cfg["model"]["img_h"], cfg["model"]["to_gray"]
    tfm = ImageTransform(mw, mh, to_gray, None)

    # builder de loaders con flags del preset
    use_offline_spikes = bool(cfg["data"]["use_offline_spikes"])
    encode_runtime     = bool(cfg["data"]["encode_runtime"])
    make_loader_fn = build_make_loader_fn(
        root=ROOT, use_offline_spikes=use_offline_spikes, encode_runtime=encode_runtime
    )

    # tasks
    with open(args.tasks_file, "r", encoding="utf-8") as f:
        tasks_json = json.load(f)
    task_list = [{"name": n, "paths": tasks_json["splits"][n]} for n in tasks_json["tasks_order"]]

    # factory de modelo
    def make_model_fn(tfm):
        return build_model(cfg["model"]["name"], tfm, beta=0.9, threshold=0.5)

    out_dir, _ = run_continual(
        task_list=task_list,
        make_loader_fn=make_loader_fn,
        make_model_fn=make_model_fn,
        tfm=tfm,
        cfg=cfg,
        preset_name=args.preset,
        out_root=ROOT/"outputs",
        verbose=True,
    )
    print("OK:", out_dir)

if __name__ == "__main__":
    main()
