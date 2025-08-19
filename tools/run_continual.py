# tools/run_continual.py
from __future__ import annotations

# --- asegurar raíz del repo en sys.path ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------

import argparse, json
from pathlib import Path
import torch

from src.runner import run_continual
from src.datasets import ImageTransform
from src.models import SNNVisionRegressor
from src.utils import make_loaders_from_csvs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=["fast","std","accurate"])
    ap.add_argument("--method", required=True, choices=["naive","ewc"])
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default="rate", choices=["rate","latency","raw","image"])
    ap.add_argument("--tasks-file", default="data/processed/tasks_balanced.json")
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--no-runtime-encode", action="store_true")
    args = ap.parse_args()

    # Carga de tareas
    tasks_json = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
    task_list = [{"name": n, "paths": tasks_json["splits"][n]} for n in tasks_json["tasks_order"]]

    # Transform por defecto
    tfm = ImageTransform(160, 80, True, None)

    # Model factory
    def make_model_fn(tfm):
        return SNNVisionRegressor(in_channels=(1 if tfm.to_gray else 3), lif_beta=0.95)

    # Loader factory QUE RESPETA runtime_encode
    runtime_encode = (not args.no_runtime_encode)
    def make_loader_fn(task, batch_size, encoder, T, gain, tfm, seed, **dl_kwargs):
        RAW = Path("data") / "raw" / "udacity" / task["name"]
        paths = task["paths"]

        return make_loaders_from_csvs(
            base_dir=RAW,
            train_csv=Path(paths["train"]),
            val_csv=Path(paths["val"]),
            test_csv=Path(paths["test"]),
            batch_size=batch_size,
            encoder=encoder,
            T=T, gain=gain, tfm=tfm, seed=seed,
            **dl_kwargs
        )

    out_dir, res = run_continual(
        task_list=task_list,
        make_loader_fn=make_loader_fn,
        make_model_fn=make_model_fn,
        tfm=tfm,
        preset=args.preset,
        method=args.method,
        lam=(args.lam if args.method == "ewc" else None),
        seed=args.seed,
        encoder=args.encoder,                         # codificador temporal deseado (para runtime)
        fisher_batches_by_preset={"fast": 200, "std": 600, "accurate": 800},
        epochs_override=args.epochs_override,
        runtime_encode=runtime_encode,
        out_root=args.out_root,
        verbose=True,
    )
    print("OK:", out_dir)

if __name__ == "__main__":
    main()
