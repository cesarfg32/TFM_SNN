# src/utils_tasks.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Union

try:
    from src.config import load_preset  # nuestra función robusta
except Exception:
    load_preset = None  # type: ignore

def _as_cfg(cfg_or_name: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(cfg_or_name, dict):
        return cfg_or_name
    if isinstance(cfg_or_name, str):
        if load_preset is None:
            raise RuntimeError(
                "build_task_list_for recibió un nombre de preset pero no puede "
                "importar load_preset; pasa el dict ya cargado o instala PyYAML."
            )
        return load_preset(cfg_or_name)
    raise TypeError(f"Tipo no soportado para cfg_or_name: {type(cfg_or_name)}")

def build_task_list_for(
    cfg_or_name: Union[str, Dict[str, Any]],
    *,
    default_runs: Iterable[str] = ("circuito1", "circuito2"),
) -> List[Dict[str, Any]]:
    cfg = _as_cfg(cfg_or_name)
    prep = dict(cfg.get("prep", {}) or {})
    runs = prep.get("runs")
    if not runs:
        runs = list(default_runs)
    return [{"name": str(r)} for r in runs]
