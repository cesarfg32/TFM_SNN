# TFM — SNN + Aprendizaje Continuo en SNN (steering Udacity)

Proyecto para investigar **aprendizaje continuo (Continual Learning, CL)** en **redes de impulsos (Spiking Neural Networks, SNN)** aplicadas a **regresión de ángulo de dirección (steering)** en conducción simulada (Udacity).
Stack: **PyTorch + snnTorch**, ejecución local en **Linux/WSL2 + CUDA** o CPU.

> **Datos**: guía y estructura en [`data/README.md`](data/README.md).  
> **Simulador (inferencia en tiempo real)**: ver [`tools/README_sim.md`](tools/README_sim.md).

---

## 1) Requisitos

- **Python 3.12** (recomendado) y `pip`.
- Linux o **WSL2 (Ubuntu 24.04)** en Windows 11.
- GPU NVIDIA opcional (recomendado). Instala la *wheel* de PyTorch para tu versión CUDA.
- No requiere compilación (todo Python).

> En WSL2, asegúrate de:
> - Driver NVIDIA actualizado en Windows.
> - Soporte CUDA visible dentro de WSL (`nvidia-smi`).

---

## 2) Instalación rápida

```bash
# Clona el repo
git clone https://github.com/cesarfg32/TFM_SNN.git
cd TFM_SNN

# Crea y activa el entorno del proyecto
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Instala PyTorch para tu GPU (ejemplo: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Resto de dependencias
pip install -r requirements.txt

# Comprobación de entorno (versiones, CUDA, GPU, estructura)
python tools/check_env.py
```

---

## 3) Organización del proyecto

```
configs/
  presets.yaml            # perfiles (fast / std / accurate)
data/
  raw/udacity/...         # datos originales (ver data/README.md)
  processed/...           # splits, balanceados y H5 generados
notebooks/
  01_DATA_QC_PREP.ipynb   # QC + normalización rutas + splits (train/val/test) + tasks.json
  01A_PREP_BALANCED.ipynb # (opcional) balanceo por bins offline + verificación
  02_ENCODE_OFFLINE.ipynb # (opcional) codificación a spikes offline (HDF5, formato v2)
  03_TRAIN_CONTINUAL.ipynb# entrenamiento continual (EWC, rehearsal, etc.) + ES opcional
  03B_HPO_OPTUNA.ipynb    # HPO básico con Optuna sobre continual
  04_RESULTS.ipynb        # agregación de resultados y figuras
outputs/
  ...                     # métricas y manifiestos por experimento
src/
  datasets.py             # Udacity CSV/H5 + augment + balanceo online
  models.py               # SNNVision / PilotNet ANN+SNN (dinámicos)
  training.py             # bucles de entrenamiento + early stopping + AMP
  runner.py               # orquestador continual (tareas, eval secuencial)
  utils.py                # seeds, presets, dataloaders factory
  bench.py                # utilidades de benchmarking
  prep/
    data_prep.py          # limpieza, splits, balanceo offline (duplicación)
    encode_offline.py     # CSV → H5 (spikes v2)
    augment_offline.py    # balanceo con imágenes aumentadas reales
tools/
  prep_udacity.py         # CLI: QC + splits (+ opcional oversampling por bins)
  prep_offline.py         # CLI: pipeline completo prep + balanceo-img + (opcional) H5
  encode_from_tasks.py    # CLI: tasks*.json → H5 (parámetros explícitos)
  encode_offline.py       # CLI: CSV → H5 (una llamada)
  encode_tasks.py         # CLI: usa presets.yaml para codificar en bloque
  run_continual.py        # CLI: entrenamiento continual con preset
  sim_drive.py            # Cliente WebSocket para el simulador Udacity (inferencia)
  README_sim.md           # Guía de integración del simulador
```

---

## 4) Flujo de trabajo

### 4.1 Preparar datos

Coloca tus recorridos del simulador Udacity así (ver `data/README.md` para más detalle):

```
data/raw/udacity/circuito1/driving_log.csv  +  data/raw/udacity/circuito1/IMG/
data/raw/udacity/circuito2/driving_log.csv  +  data/raw/udacity/circuito2/IMG/
```

Ejecuta **01_DATA_QC_PREP.ipynb** (kernel de `.venv`). Genera:
- `data/processed/<run>/canonical.csv`
- `data/processed/<run>/{train,val,test}.csv`
- `data/processed/tasks.json` con el orden de tareas (p. ej. `["circuito1","circuito2"]`).

> *(Opcional)* **01A_PREP_BALANCED.ipynb** genera `train_balanced.csv` por bins (oversampling) y gráficos de histograma.

> *(Opcional)* **02_ENCODE_OFFLINE.ipynb** crea HDF5 con spikes si quieres depurar E/S o medir diferencias offline vs on‑the‑fly.

### 4.2 Entrenamiento continual

Abre **03_TRAIN_CONTINUAL.ipynb** y ejecuta:
- **Ejecución base** con preset `fast` para verificar la tubería.
- **Comparativa de métodos**: `naive`, `ewc`, `rehearsal`, `rehearsal+ewc` (los dos últimos si están habilitados en `src/methods/`).  
  Se guardan métricas en `outputs/` y un `continual_results.json` por run.

*(Opcional)* **03B_HPO_OPTUNA.ipynb** lanza estudios Optuna sobre hiperparámetros de los métodos CL.

---

## 5) Presets y configuración

Archivo: `configs/presets.yaml`. Ejemplo mínimo (esquema actual):

```yaml
fast:
  model:
    name: pilotnet_snn      # o: pilotnet_ann | snn_vision
    img_w: 200
    img_h: 66
    to_gray: true

  data:
    encoder: rate           # rate | latency | raw | image
    T: 20
    gain: 0.5               # solo 'rate'
    seed: 42

    # Fuente de datos y pipeline
    use_offline_spikes: false   # usar H5 (spikes) si existen
    encode_runtime: true        # si el loader entrega (B,C,H,W), codifica en GPU a (T,B,C,H,W)
    use_offline_balanced: false # si existe tasks_balanced.json, úsalo

    # DataLoader
    num_workers: 8
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4

    # Augment (solo train; opcional)
    aug_train: { prob_hflip: 0.3, brightness: [0.8, 1.2] }

    # Balanceo online (entrenamiento)
    balance_online: false
    # Si activas balance_online, puedes ajustar (o dejar None para usar defaults internos):
    balance_bins: 21
    balance_smooth_eps: 0.001

  optim:
    epochs: 8
    batch_size: 32
    lr: 1e-3
    amp: true                # mixed precision si hay CUDA

  continual:
    method: ewc              # naive | ewc | rehearsal | rehearsal+ewc
    params: { lam: 7e8, fisher_batches: 800 }

  naming:
    tag: ""                  # etiqueta opcional para carpetas de salida
```

**Notas**

- Si `balance_online: true` y en el preset pones `balance_bins: null` o `balance_smooth_eps: null`,
  el código usa por defecto `21` y `1e-3`. Evita errores de `NoneType → int/float` en notebooks.
- `use_offline_spikes` y `encode_runtime` no deben estar **ambos** a `true` a la vez.

---

## 6) Resultados (`outputs/`)

Cada ejecución crea una carpeta como:

```
outputs/
  continual_fast_ewc_rate_model-PilotNetSNN_66x200_gray_seed_42/
    task_1_circuito1/manifest.json
    task_2_circuito2/manifest.json
    continual_results.json    # métricas por tarea y olvido
```

En `manifest.json` de entrenamiento guardamos:
- Historial `train_loss` / `val_loss`.
- Early stopping (si se activa en el runner/notebook).
- Tiempos y dispositivo.

---

## 7) Scripts útiles (CLI)

- **QC + splits (tasks.json)**  
  ```bash
  python tools/prep_udacity.py --root . --runs circuito1 circuito2 --use-left-right --steer-shift 0.2 --bins 21 --train 0.70 --val 0.15 --seed 42
  ```

- **Pipeline prep completo (incluye balanceo con imágenes y, opcional, H5)**  
  ```bash
  python tools/prep_offline.py --preset fast --config configs/presets.yaml --encode
  ```

- **Codificar tasks*.json → H5**  
  ```bash
  python tools/encode_from_tasks.py --tasks-file data/processed/tasks.json --encoder rate --T 20 --gain 0.5 --w 200 --h 66 --seed 42 --only-missing
  ```

- **Entrenamiento continual con preset**  
  ```bash
  python tools/run_continual.py --preset fast --config configs/presets.yaml --tasks-file data/processed/tasks.json --tag prueba1
  ```

- **Inferencia en simulador** (tras entrenar y exportar `model_best.pt`)  
  ```bash
  python tools/sim_drive.py --host 127.0.0.1 --port 4567       --model-path outputs/continual_fast_ewc_rate_model-PilotNetSNN_66x200_gray_seed_42/model_best.pt       --model-name pilotnet_snn --img-w 200 --img-h 66 --to-gray       --encoder rate --T 20 --gain 0.5
  ```

Más detalle en [`tools/README_sim.md`](tools/README_sim.md).

---

## 8) Resolución de problemas

- **CUDA no detectada** → revisa drivers y reinstala PyTorch con la *wheel* correcta.
- **`FileNotFoundError: data/processed/tasks.json`** → ejecuta primero los notebooks de preparación.
- **OOM (VRAM insuficiente)** → usa preset `fast`, reduce `batch_size` o `T`, y deja `amp: true`.
- **Kernel de notebooks incorrecto** → selecciona el intérprete de `.venv` en VS Code.

---

## 9) Estado de métodos CL

- **EWC** implementado e integrado.
- **Rehearsal** básico disponible si lo activas en `METHODS` (notebooks).
- Stubs de variantes (SA‑SNN / AS‑SNN / SCA‑SNN / CoLaNET) en `src/methods/` para desarrollo futuro.

---

¿Dudas? Abre un issue o comenta qué método te interesa seguir desarrollando.
