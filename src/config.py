# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Iterable, List
import os
import json

try:
    import yaml  # PyYAML
except Exception as e:
    raise ImportError("Falta PyYAML. Instálalo con: pip install pyyaml") from e


# =========================
#  Búsqueda de presets
# =========================
_PRESETS_DIR_ENV   = "PRESETS_DIR"   # para modelo "un preset por archivo"
_PRESETS_FILE_ENV  = "PRESETS_FILE"  # para "colección" (varios presets en 1 YAML)

# Directorios por defecto donde buscar ficheros individuales y/o colección
_DEFAULT_DIRS: tuple[Path, ...] = (
    Path("."),                 # raíz del repo
    Path("configs/presets"),
    Path("config/presets"),
    Path("configs"),
    Path("config"),
)

# Candidatos de archivo "colección" (varios presets en un mismo YAML)
_DEFAULT_COLLECTION_BASENAMES: tuple[str, ...] = (
    "presets.yaml",
    "presets.yml",
)


def _iter_candidate_dirs() -> Iterable[Path]:
    """Genera carpetas a explorar, respetando PRESETS_DIR si existe."""
    env = os.getenv(_PRESETS_DIR_ENV, None)
    if env:
        p = Path(env)
        if p.exists() and p.is_dir():
            yield p
    for d in _DEFAULT_DIRS:
        if d.exists() and d.is_dir():
            yield d


def _iter_collection_files() -> Iterable[Path]:
    """Genera archivos colección candidatos, respetando PRESETS_FILE si existe."""
    env_file = os.getenv(_PRESETS_FILE_ENV, None)
    if env_file:
        p = Path(env_file)
        if p.exists() and p.is_file():
            yield p
    for d in _iter_candidate_dirs():
        for base in _DEFAULT_COLLECTION_BASENAMES:
            p = d / base
            if p.exists() and p.is_file():
                yield p


def _find_individual_preset_file(name: str) -> Optional[Path]:
    """
    Busca 'name.yaml' o 'name.yml' en las carpetas conocidas.
    Devuelve Path si lo encuentra; si no, None (no dispara excepción aquí).
    """
    for d in _iter_candidate_dirs():
        for suf in (".yaml", ".yml"):
            cand = d / f"{name}{suf}"
            if cand.exists():
                return cand
    return None


def _yaml_load(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("El YAML debe deserializar a dict (mapping).")
        return data
    except Exception as e:
        raise RuntimeError(f"Error al leer YAML de {path}: {e}") from e


def _deep_update(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Actualiza recursivamente 'base' con 'overlay' (dict profundo)."""
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _ensure_sections(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asegura presencia de secciones mínimas esperadas por runner/training:
    data, optim, continual, naming, logging, prep, model. No pisa valores existentes.
    """
    cfg.setdefault("data", {})
    cfg.setdefault("optim", {})
    cfg.setdefault("continual", {})
    cfg.setdefault("naming", {})
    cfg.setdefault("logging", {})
    cfg.setdefault("prep", {})
    cfg.setdefault("model", {})  # NUEVO: asegura sección model

    # Defaults de data
    d = cfg["data"]
    d.setdefault("encoder", "rate")
    d.setdefault("T", 10)
    d.setdefault("gain", 0.5)
    d.setdefault("use_offline_spikes", False)
    d.setdefault("num_workers", 8)
    d.setdefault("prefetch_factor", 2)
    d.setdefault("pin_memory", True)
    d.setdefault("persistent_workers", True)

    # Defaults de optim
    o = cfg["optim"]
    o.setdefault("epochs", 2)
    o.setdefault("batch_size", 8)
    o.setdefault("lr", 1e-3)
    o.setdefault("amp", True)
    o.setdefault("es_patience", None)
    o.setdefault("es_min_delta", 0.0)

    # Defaults de continual
    c = cfg["continual"]
    c.setdefault("method", "naive")
    c.setdefault("params", {})

    # Defaults de model (alineados con utils_components)
    m = cfg["model"]
    m.setdefault("name", "pilotnet_snn")
    m.setdefault("img_w", 200)
    m.setdefault("img_h", 66)
    m.setdefault("to_gray", True)

    return cfg


def _resolve_from_collection(key: str) -> Optional[tuple[Path, Dict[str, Any]]]:
    """
    Intenta encontrar un archivo 'colección' que contenga una clave top-level == key
    (p.ej., 'fast', 'std', 'accurate') y la devuelve.
    """
    for file in _iter_collection_files():
        data = _yaml_load(file)
        # Caso 1: el YAML tiene directamente fast/std/accurate como claves top-level
        if key in data and isinstance(data[key], dict):
            return file, data[key]
        # Caso 2: el YAML anida bajo 'presets': { fast: {...}, ... }
        presets = data.get("presets")
        if isinstance(presets, dict) and key in presets and isinstance(presets[key], dict):
            return file, presets[key]
    return None


def _load_extends_token(token: str) -> Dict[str, Any]:
    """
    Carga un preset indicado en 'extends'. Soporta:
      - Nombre simple: 'fast'
      - Ruta YAML completa: 'configs/fast.yaml'
      - Ruta + clave: 'configs/presets.yaml:fast'
    """
    token = str(token).strip()
    if ":" in token:
        left, right = token.split(":", 1)
        p = Path(left)
        if p.suffix.lower() in (".yaml", ".yml"):
            return load_preset(p, right)
    # Si no hay ':', o no es YAML, deja que load_preset resuelva nombre o ruta
    return load_preset(token)


def _apply_extends_chain(base_cfg: Dict[str, Any], extends_decl: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Aplica uno o varios 'extends' sobre base_cfg (overlay por la derecha).
    extends_decl puede ser string o lista de strings.
    """
    if isinstance(extends_decl, str):
        parent = _load_extends_token(extends_decl)
        return _deep_update(parent, base_cfg)
    elif isinstance(extends_decl, list):
        merged: Dict[str, Any] = {}
        for tok in extends_decl:
            parent = _load_extends_token(tok)
            _deep_update(merged, parent)
        return _deep_update(merged, base_cfg)
    else:
        return base_cfg


def load_preset(
    name_or_path: Union[str, os.PathLike],
    preset_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Carga un preset.

    Usos soportados:
      1) load_preset("std")
         - Busca 'std' en archivos individuales o en una colección (p.ej. presets.yaml).

      2) load_preset("configs/presets.yaml", "std")
         - Carga el YAML indicado y devuelve la subclave 'std' (o bajo 'presets.std').

      3) load_preset(Path("configs/presets.yaml"), "std")
         - Igual que (2), usando Path.

      4) load_preset("configs/mi_preset_individual.yaml")
         - Carga un YAML individual completo (sin subclave).

    Además:
      - Soporta herencia con 'extends' (string o lista):
          extends: fast
          extends: [fast, configs/otra.yaml:clave]
    Devuelve un dict con secciones mínimas aseguradas y meta-información en _meta.
    """
    p = Path(str(name_or_path))

    # --- Modo 2-parámetros: ruta YAML + clave ---
    if preset_key is not None:
        if p.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError(
                "Cuando se pasa 'preset_key', el primer argumento debe ser una ruta a YAML "
                f"(recibido: {p})"
            )
        # Resolver ruta relativa al CWD si fuera necesario
        if not p.exists():
            p2 = Path.cwd() / p
            if p2.exists():
                p = p2
            else:
                raise FileNotFoundError(f"No existe el archivo de preset: {p}")

        data = _yaml_load(p)

        # Buscar clave directa
        if preset_key in data and isinstance(data[preset_key], dict):
            subcfg = data[preset_key]
        else:
            # Buscar bajo 'presets'
            presets = data.get("presets")
            if isinstance(presets, dict) and preset_key in presets and isinstance(presets[preset_key], dict):
                subcfg = presets[preset_key]
            else:
                raise KeyError(
                    f"El YAML {p} no contiene la clave '{preset_key}' ni bajo 'presets'."
                )

        # Herencia si el subbloque define 'extends'
        ext = subcfg.pop("extends", None)
        if ext:
            subcfg = _apply_extends_chain(subcfg, ext)

        cfg = _ensure_sections(subcfg)
        cfg.setdefault("_meta", {})
        cfg["_meta"]["preset_path"] = str(p.resolve())
        cfg["_meta"]["preset_key"] = str(preset_key)
        return cfg

    # --- Modo 1-parámetro ---
    # 1) Si te pasan una ruta a YAML, úsala directamente
    if p.suffix.lower() in (".yaml", ".yml"):
        if not p.exists():
            p2 = Path.cwd() / p
            if p2.exists():
                p = p2
            else:
                raise FileNotFoundError(f"No existe el archivo de preset: {p}")

        data = _yaml_load(p)

        # Herencia si el YAML top-level define 'extends'
        ext = data.pop("extends", None)
        if ext:
            data = _apply_extends_chain(data, ext)

        cfg = _ensure_sections(data)
        cfg.setdefault("_meta", {})
        cfg["_meta"]["preset_path"] = str(p.resolve())
        return cfg

    # 2) Nombre -> archivo individual
    found = _find_individual_preset_file(p.name)
    if found is not None:
        data = _yaml_load(found)

        # Herencia si define 'extends'
        ext = data.pop("extends", None)
        if ext:
            data = _apply_extends_chain(data, ext)

        cfg = _ensure_sections(data)
        cfg.setdefault("_meta", {})
        cfg["_meta"]["preset_path"] = str(found.resolve())
        return cfg

    # 3) Nombre -> dentro de colección (presets.yaml)
    resolved = _resolve_from_collection(p.name)
    if resolved is not None:
        file, subcfg = resolved

        # Herencia si el bloque define 'extends'
        ext = subcfg.pop("extends", None)
        if ext:
            subcfg = _apply_extends_chain(subcfg, ext)

        cfg = _ensure_sections(subcfg)
        cfg.setdefault("_meta", {})
        cfg["_meta"]["preset_path"] = str(file.resolve())
        cfg["_meta"]["preset_key"] = p.name
        return cfg

    # 4) Nada encontrado
    search_places = [str(d) for d in _iter_candidate_dirs()]
    raise FileNotFoundError(
        f"No se encontró el preset '{name_or_path}'.\n"
        f"- Coloca 'presets.yaml' en: {', '.join(search_places)} (o usa {_PRESETS_FILE_ENV})\n"
        f"- O crea 'configs/{p.name}.yaml' (o usa {_PRESETS_DIR_ENV})\n"
        f"- O pasa la ruta completa a un YAML."
    )


def dump_cfg(cfg: Dict[str, Any]) -> str:
    try:
        return json.dumps(cfg, indent=2, ensure_ascii=False)
    except Exception:
        return str(cfg)
