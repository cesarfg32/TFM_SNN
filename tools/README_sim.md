# Guía de integración con el simulador Udacity (`tools/sim_drive.py`)

Este documento explica cómo ejecutar **inferencia en tiempo real** contra el *Udacity Self‑Driving Car Simulator* (modo *Autonomous*), conectando nuestro modelo por **WebSocket**.

> Para entrenar/exportar el modelo consulta el README general y los notebooks `03_TRAIN_CONTINUAL.ipynb` / `03B_HPO_OPTUNA.ipynb`.

---

## 1) Requisitos

- Python 3.12 activado (`.venv` del proyecto).
- Paquetes ya instalados con `pip install -r requirements.txt`.
- Simulador Udacity abierto en modo **Autonomous** (escenario *Track 1* o *Track 2*).
- Modelo exportado como `model_best.pt` (o `last`) en `outputs/...`.

---

## 2) Exportar/guardar el modelo

El `training` ya guarda en cada epoch:
- `model_last.pt` → último estado
- `model_best.pt` → mejor `val_loss` (si hubo mejora; si no, se duplica `last` como `best`)

Si tienes un *checkpoint* que guarda un objeto más complejo (`{"model": nn.Module, ...}`) y quieres **solo** el `state_dict`, convierte con:

```python
import torch
obj = torch.load("modelo_completo.pt", map_location="cpu")
state = obj["model"].state_dict() if isinstance(obj, dict) and "model" in obj else obj.state_dict()
torch.save(state, "model_best.pt")
```

---

## 3) Arrancar el simulador

1. Abre el simulador (Windows/Mac) y selecciona **Autonomous mode**.
2. Deja el coche en pista (Track 1 o 2). El simulador abre un servidor WS en `ws://127.0.0.1:4567`.

---

## 4) Lanzar el cliente (`sim_drive.py`)

Ejemplo (PilotNet SNN en escala de grises 200×66, codificación `rate` en GPU):

```bash
python tools/sim_drive.py --host 127.0.0.1 --port 4567   --model-path outputs/continual_fast_ewc_rate_model-PilotNetSNN_66x200_gray_seed_42/model_best.pt   --model-name pilotnet_snn --img-w 200 --img-h 66 --to-gray   --encoder rate --T 20 --gain 0.5   --steer-scale 1.0 --steer-clip 1.0 --fps-log 2.0
```

Argumentos importantes:
- `--model-name` ∈ `{pilotnet_snn, pilotnet_ann, snn_vision}` (de `src/models.py`).
- `--img-w --img-h --to-gray` deben **coincidir** con el *training*.
- `--encoder`/`--T`/`--gain` deben coincidir si tu modelo espera secuencias `(T,B,C,H,W)`.
  - Si el *checkpoint* fue entrenado con frames ya codificados (H5), pero aquí recibes imágenes,
    activa la codificación **en GPU** con `--encoder rate|latency|raw` y los `--T / --gain` correctos.
- `--steer-scale` y `--steer-clip` para ajustar la magnitud de salida si el coche zigzaguea.

---

## 5) Cómo funciona

- El simulador envía imágenes `base64` + `telemetría` por WebSocket.
- `sim_drive.py`:
  1) decodifica la imagen y la normaliza a `[0,1]`,
  2) aplica `ImageTransform(w,h,to_gray)`,
  3) **si procede**, codifica en GPU a `(T,B,C,H,W)` (rate/latency/raw) para SNN,
  4) ejecuta el modelo en `torch.inference_mode()` + AMP (si CUDA),
  5) devuelve el `steering` (y opcionalmente `throttle`/`brake` fijo).

> Por defecto se envía un `throttle` constante; puedes exponer un control más sofisticado si lo necesitas.

---

## 6) Consejos de estabilidad

- Empieza con `--steer-scale 0.6~0.8` si el coche hace *S‑curves*.
- Activa `--to-gray` si entrenaste en grises (1 canal).
- Si el FPS cae, baja `--T` (si usas `rate/latency`) o usa `pilotnet_ann` como *smoke test*.
- Revisa la consola: se reporta `FPS` medio y latencia en ms/bin.

---

## 7) Comandos comunes

- PilotNet ANN (sin codificación temporal):
  ```bash
  python tools/sim_drive.py --host 127.0.0.1 --port 4567     --model-path outputs/supervised_fast_naive_image_model-PilotNetANN_66x200_gray_seed_42/model_best.pt     --model-name pilotnet_ann --img-w 200 --img-h 66 --to-gray     --encoder image --steer-scale 0.8
  ```

- PilotNet SNN con `latency`:
  ```bash
  python tools/sim_drive.py --host 127.0.0.1 --port 4567     --model-path outputs/continual_std_ewc_latency_model-PilotNetSNN_66x200_gray_seed_42/model_best.pt     --model-name pilotnet_snn --img-w 200 --img-h 66 --to-gray     --encoder latency --T 20 --steer-scale 0.8
  ```

---

## 8) Troubleshooting

- **`ConnectionRefusedError`** → ¿Estás en modo *Autonomous*? ¿Puerto 4567 correcto?
- **El coche vibra/oscila** → baja `--steer-scale`, o sube `--steer-clip` ligeramente.
- **Imagen desbordada** → revisa `--img-w/--img-h/--to-gray` y que coincidan con entrenamiento.
- **CUDA OOM** → usa `pilotnet_ann` (sin temporal) o reduce `--T`/resolución.
- **Mismatch de encoder** → alinea `--encoder/--T/--gain` con el entrenamiento.

---

## 9) Desarrollo

El archivo `tools/sim_drive.py` está organizado en:
- Carga del modelo + `build_model` (de `src/models.py`).
- Preprocesado (`ImageTransform` de `src/datasets.py`).
- Codificación opcional en GPU con `set_encode_runtime()` (de `src/training.py`).
- *Loop* WebSocket con `websocket-client` y JSON minimalista.

Si quieres registrar vídeo o *dashboards*, añade *hooks* donde se construye el tensor de entrada o al emitir la predicción.
