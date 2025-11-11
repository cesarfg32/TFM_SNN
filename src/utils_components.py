# src/utils_components.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Tuple

from src.utils import build_make_loader_fn  # tu util original
from src.models import build_model  # <-- usa tu propio factory

@dataclass
class _TFMShim:
    h: int
    w: int
    to_gray: bool

def _snake_to_camel(s: str) -> str:
    return "".join(part.capitalize() for part in s.replace("-", "_").split("_"))

def _guess_repo_root(cfg) -> Path:
    """
    Intenta derivar la raíz del repo a partir del path del preset.
    Si presets.yaml está en <repo>/configs/presets.yaml, la raíz es <repo>.
    """
    meta = cfg.get("_meta", {}) or {}
    p = meta.get("preset_path")
    if p:
        pp = Path(p).resolve()
        # directorio que contiene el presets.yaml (ej. <repo>/configs)
        d = pp.parent
        # preferimos su padre si existe <repo>/data/processed
        cand = d.parent
        if (cand / "data" / "processed").exists():
            return cand
        # si no, prueba con d (por si el preset estuviera justo en <repo>)
        if (d / "data" / "processed").exists():
            return d
    # fallback a CWD o su padre si ahí está data/processed
    cwd = Path.cwd()
    for c in (cwd, cwd.parent, cwd.parent.parent):
        if (c / "data" / "processed").exists():
            return c
    # último recurso: cwd
    return cwd

def _pick_loader_root(cfg) -> Path:
    """
    Devuelve la carpeta RAÍZ que espera build_make_loader_fn (no la carpeta data),
    es decir, aquella tal que <root>/data/processed exista.
    """
    r = _guess_repo_root(cfg)
    if (r / "data" / "processed").exists():
        return r
    # intenta un par de alternativas razonables
    for alt in (r.parent, r.parent.parent, Path.cwd(), Path.cwd().parent):
        if (alt / "data" / "processed").exists():
            return alt
    # si no existe, devolvemos r igualmente para no romper la llamada
    return r

def _normalize_model_name(raw: str) -> str:
    # Acepta CamelCase, guiones y nombres sin guiones bajos:
    s = raw.strip()
    # alias CamelCase frecuentes
    if s == "PilotNetSNN":
        return "pilotnet_snn"
    if s == "PilotNetANN":
        return "pilotnet_ann"

    s = s.lower().replace("-", "_")
    s_no = s.replace("_", "")

    # alias sin guión bajo
    if s_no == "pilotnetsnn":
        return "pilotnet_snn"
    if s_no == "pilotnetann":
        return "pilotnet_ann"
    if s_no == "snnvision":
        return "snn_vision"

    return s  # deja lo que venga si ya es canónico

def _make_model_factory(model_cfg: dict):
    raw = str(model_cfg.get("name", "pilotnet_snn"))
    name = _normalize_model_name(raw)

    def make_model_fn(tfm):
        return build_model(name, tfm)

    return make_model_fn

def build_components_for(cfg) -> Tuple[Callable, Callable, Any]:
    data = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    img_h = int(model_cfg.get("img_h", data.get("img_h", 66)))
    img_w = int(model_cfg.get("img_w", data.get("img_w", 200)))
    to_gray = bool(model_cfg.get("to_gray", data.get("to_gray", True)))
    tfm = _TFMShim(h=img_h, w=img_w, to_gray=to_gray)

    # CLAVE: pasar la RAÍZ del repo (donde cuelga data/processed), no <repo>/data
    loader_root = _pick_loader_root(cfg)

    use_offline = bool(data.get("use_offline_spikes", False))
    encode_runtime = bool(data.get("encode_runtime", not use_offline))

    make_loader_fn = build_make_loader_fn(
        root=loader_root,              # <- aquí va la raíz del repo
        use_offline_spikes=use_offline,
        encode_runtime=encode_runtime,
    )
    make_model_fn = _make_model_factory(model_cfg)
    return make_loader_fn, make_model_fn, tfm
