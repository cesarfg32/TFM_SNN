#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprobación de entorno para TFM SNN-CL.

- Muestra versiones de Python/PyTorch/torchvision/snnTorch/OpenCV/Numpy/Pandas.
- Comprueba CUDA y la GPU disponible.
- Ejecuta una operación simple en GPU si está disponible (matmul pequeña).
- Verifica que existen carpetas/archivos clave del proyecto.
"""

from pathlib import Path
import sys, time

def main():
    root = Path(__file__).resolve().parents[1]
    print("== Entorno del TFM SNN-CL ==")
    print(f"Ruta del proyecto: {root}")

    # Python
    print(f"[OK] Python: {sys.version.split()[0]}")

    # Paquetes
    def ver(mod, attr="__version__"):
        try:
            m = __import__(mod)
            v = getattr(m, attr, "unknown")
            print(f"[OK] {mod}: {v}")
            return m
        except Exception as e:
            print(f"[WARN] No se pudo importar {mod}: {e}")
            return None

    torch = ver("torch")
    tv = ver("torchvision")
    snn = ver("snntorch")
    cv2 = ver("cv2")
    np = ver("numpy")
    pd = ver("pandas")

    # CUDA / GPU
    if torch:
        try:
            print(f"[INFO] torch.version.cuda: {torch.version.cuda}")
            avail = torch.cuda.is_available()
            print(f"[OK] CUDA disponible: {avail}")
            if avail:
                dev_name = torch.cuda.get_device_name(0)
                print(f"[OK] GPU: {dev_name}")
                # Smoke op en GPU
                import torch as T
                a = T.randn((512, 512), device="cuda")
                b = T.randn((512, 512), device="cuda")
                t0 = time.time()
                c = a @ b
                T.cuda.synchronize()
                dt = (time.time() - t0) * 1000
                print(f"[OK] Matmul 512x512 en GPU: {dt:.2f} ms")
            else:
                print("[INFO] Sin GPU/CUDA: se usará CPU.")
        except Exception as e:
            print(f"[WARN] Prueba CUDA fallida: {e}")

    # Estructura del proyecto
    must_exist = [
        root/"src", root/"notebooks", root/"configs"/"presets.yaml", root/".gitignore",
        root/"requirements.txt"
    ]
    for p in must_exist:
        print(f"[OK] Existe: {p}") if p.exists() else print(f"[WARN] Falta: {p}")

    # Datos (informativo)
    raw = root/"data"/"raw"/"udacity"
    if raw.exists():
        print(f"[OK] Datos RAW: {raw}")
    else:
        print(f"[INFO] Aún no hay datos en: {raw}")

    print("== Fin de comprobación ==")

if __name__ == "__main__":
    main()