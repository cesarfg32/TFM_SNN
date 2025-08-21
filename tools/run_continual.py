# tools/run_continual.py
from __future__ import annotations

# --- asegurar raíz del repo en sys.path ---
import sys, argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------

from src.runner import run_continual
from src.datasets import ImageTransform
from src.models import build_model, default_tfm_for_model
from src.utils import make_loaders_from_csvs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=["fast", "std", "accurate"])
    ap.add_argument("--method", required=True,
                    choices=["naive", "ewc", "rehearsal", "rehearsal+ewc"])
    # Hiperparámetros (se mapean a method_kwargs):
    ap.add_argument("--lam", type=float, default=None, help="λ para EWC")
    ap.add_argument("--ewc-lam", type=float, default=None,
                    help="λ de EWC cuando --method rehearsal+ewc")
    ap.add_argument("--buffer-size", type=int, default=10_000)
    ap.add_argument("--replay-ratio", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default="rate",
                    choices=["rate", "latency", "raw", "image"])
    ap.add_argument("--tasks-file", default="data/processed/tasks_balanced.json")
    ap.add_argument("--epochs-override", type=int, default=None)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--no-runtime-encode", action="store_true")

    ap.add_argument("--model", default="snn_vision",
                    choices=["snn_vision", "pilotnet_ann", "pilotnet_snn"])
    ap.add_argument("--img-w", type=int, default=None)
    ap.add_argument("--img-h", type=int, default=None)
    ap.add_argument("--rgb", action="store_true")
    args = ap.parse_args()

    # Carga de tareas
    tasks_json = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
    task_list = [{"name": n, "paths": tasks_json["splits"][n]}
                 for n in tasks_json["tasks_order"]]

    # Transform por defecto o forzado
    if args.img_w is None or args.img_h is None:
        tfm = default_tfm_for_model(args.model, to_gray=(not args.rgb))
    else:
        tfm = ImageTransform(args.img_w, args.img_h, to_gray=(not args.rgb), crop_top=None)

    # Factory del modelo
    def make_model_fn(tfm):
        return build_model(args.model, tfm, beta=0.9, threshold=0.5)

    # Loader factory que respeta runtime encode
    runtime_encode = (not args.no_runtime_encode)

    def make_loader_fn(task, batch_size, encoder, T, gain, tfm, seed, **dl_kwargs):
        RAW = Path("data") / "raw" / "udacity" / task["name"]
        paths = task["paths"]
        # Si runtime-encode está ON y el encoder temporal es rate/latency/raw, pedimos 4D (image)
        encoder_for_loader = "image" if (runtime_encode and encoder in {"rate", "latency", "raw"}) else encoder
        return make_loaders_from_csvs(
            base_dir=RAW,
            train_csv=Path(paths["train"]),
            val_csv=Path(paths["val"]),
            test_csv=Path(paths["test"]),
            batch_size=batch_size,
            encoder=encoder_for_loader,
            T=T, gain=gain, tfm=tfm, seed=seed,
            **dl_kwargs
        )

    # Mapear flags -> method_kwargs
    method_kwargs = {}
    if args.method == "ewc":
        if args.lam is None:
            ap.error("--lam es obligatorio cuando --method ewc")
        method_kwargs["lam"] = args.lam

    elif args.method == "rehearsal":
        method_kwargs.update({
            "buffer_size": args.buffer_size,
            "replay_ratio": args.replay_ratio,
        })

    elif args.method == "rehearsal+ewc":
        if args.ewc_lam is None:
            ap.error("--ewc-lam es obligatorio cuando --method rehearsal+ewc")
        method_kwargs.update({
            "buffer_size": args.buffer_size,
            "replay_ratio": args.replay_ratio,
            "ewc_lam": args.ewc_lam,
        })
    # naive → method_kwargs = {}

    out_dir, res = run_continual(
        task_list=task_list,
        make_loader_fn=make_loader_fn,
        make_model_fn=make_model_fn,
        tfm=tfm,
        preset=args.preset,
        method=args.method,
        seed=args.seed,
        encoder=args.encoder,
        fisher_batches_by_preset={"fast": 200, "std": 600, "accurate": 800},
        epochs_override=args.epochs_override,
        runtime_encode=runtime_encode,
        out_root=args.out_root,
        verbose=True,
        method_kwargs=method_kwargs,   # << único canal de hiperparámetros
    )
    print("OK:", out_dir)

if __name__ == "__main__":
    main()
