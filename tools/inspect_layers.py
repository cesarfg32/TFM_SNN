# tools/inspect_layers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re
import torch
from typing import List, Tuple, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_preset
from src.utils_components import build_components_for

def _param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p is not None)

def _matches_types(m: torch.nn.Module, wanted: List[str]) -> bool:
    if not wanted:
        return True
    tname = type(m).__name__
    return any(tname == w or tname.lower() == w.lower() for w in wanted)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, help="Ruta a YAML o nombre de preset.")
    ap.add_argument("--filter", default="", help="Substring (regex simple) para filtrar por nombre.")
    ap.add_argument("--only-types", nargs="*", default=[], help="Limitar a tipos (p.ej., Conv2d Linear)")
    ap.add_argument("--dump-shapes", action="store_true", help="Registrar shapes de salida con forward de prueba.")
    ap.add_argument("--batch", type=int, default=1, help="Batch size para el forward de prueba.")
    args = ap.parse_args()

    cfg = load_preset(args.preset)
    make_loader_fn, make_model_fn, tfm = build_components_for(cfg)
    model = make_model_fn(tfm).eval()

    filt = args.filter
    only_types = args.only_types

    hooks = []
    out_shapes: Dict[str, Tuple[int, ...]] = {}

    def _mk_hook(name: str):
        def _h(module, inputs, output):
            t = None
            if isinstance(output, torch.Tensor):
                t = output
            elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                t = output[0]
            elif isinstance(output, dict):
                for k in ("out", "output", "logits", "y_hat", "pred"):
                    v = output.get(k, None)
                    if isinstance(v, torch.Tensor):
                        t = v; break
            if t is not None:
                out_shapes[name] = tuple(t.shape)
        return _h

    if args.dump_shapes:
        for n, m in model.named_modules():
            if n == "" or len(list(m.children())) > 0:
                continue
            if filt and re.search(filt, n, re.I) is None:
                continue
            if not _matches_types(m, only_types):
                continue
            hooks.append(m.register_forward_hook(_mk_hook(n), with_kwargs=False))

    rows = []
    for n, m in model.named_modules():
        # solo módulos "hoja"
        if n == "" or len(list(m.children())) > 0:
            continue
        if filt and re.search(filt, n, re.I) is None:
            continue
        if not _matches_types(m, only_types):
            continue
        rows.append((n, type(m).__name__, _param_count(m)))

    rows.sort(key=lambda x: x[0])
    print(f"{'name':60s}  {'type':18s}  {'params':>10s}  {'out_shape' if args.dump_shapes else ''}")
    print("-"*100)
    for n, t, p in rows:
        sh = out_shapes.get(n, None)
        print(f"{n:60s}  {t:18s}  {p:10d}  {str(sh) if sh else ''}")

    # Forward de prueba para obtener shapes si se pidió
    if args.dump_shapes:
        try:
            # Deducimos H, W y canales desde el preset/model
            h = int(cfg["model"].get("img_h", getattr(tfm, "h", 66)))
            w = int(cfg["model"].get("img_w", getattr(tfm, "w", 200)))
            to_gray = bool(cfg["model"].get("to_gray", True))
            C = 1 if to_gray else 3
            x = torch.randn(args.batch, C, h, w)
            with torch.no_grad():
                _ = model(x)
            print("\n[OK] Shapes capturadas. Si no se imprimieron arriba, repite el listado.")
        finally:
            for h in hooks:
                try: h.remove()
                except Exception: pass

if __name__ == "__main__":
    main()
