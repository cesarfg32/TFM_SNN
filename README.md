# TFM — Aprendizaje Continuo con SNN para *steering* en conducción simulada (Udacity)

Proyecto para investigar **aprendizaje continuo** (Continual Learning, **CL**) en **redes de impulsos** (Spiking Neural Networks, **SNN**) aplicado a **regresión del ángulo de dirección (*steering*)** en el simulador de Udacity.  
Stack: **PyTorch + snnTorch**, ejecución local en **Linux/WSL2 + CUDA** o CPU.

> **Datos**: guía y estructura en [`data/README.md`](data/README.md).  
> **Simulador (inferencia en tiempo real)**: ver [`tools/README_sim.md`](tools/README_sim.md).

---

## 1) Requisitos

- **Python 3.12** (recomendado) y `pip`.
- Linux o **WSL2 (Ubuntu 24.04)** en Windows 11.
- GPU NVIDIA opcional (recomendado). Instala la *wheel* de PyTorch acorde a tu versión de CUDA.
- No se requiere compilación (todo en Python).

> En WSL2:
> - Asegúrate de tener el **driver NVIDIA** actualizado en Windows.
> - Verifica CUDA dentro de WSL2 con `nvidia-smi`.

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
  presets.yaml            # perfiles declarativos (fast / std / accurate)

data/
  raw/udacity/...         # datos originales (ver data/README.md)
  processed/...           # splits, balanceados y H5 generados

notebooks/
  01_DATA_QC_PREP.ipynb   # QC + normalización + splits (train/val/test) + tasks.json
  01A_PREP_BALANCED.ipynb # (autónomo) SPLITS + (opcional) balanceo por imágenes + EDA
  02_ENCODE_OFFLINE.ipynb # (opcional) codificación a spikes offline (HDF5)
  03_TRAIN_CONTINUAL.ipynb# entrenamiento continual (métodos CL) + métricas/olvido
  03B_HPO_OPTUNA.ipynb    # HPO con Optuna (objetivo compuesto)
  04_RESULTS.ipynb        # agregación de resultados y figuras
  99_SIM_INTEGRATION.ipynb# integración con el simulador (ciclo cerrado, codificación online)

outputs/
  ...                     # manifiestos y métricas por experimento

src/
  datasets.py             # Udacity CSV/H5 + augment + balanceo online
  models.py               # backbones ANN/SNN
  runner.py               # orquestador continual (secuencia de tareas)
  utils.py                # seeds, presets, factories

  methods/                # métodos de Aprendizaje Continuo (CL)
    __init__.py
    api.py                # interfaz común (Strategy)
    registry.py           # registro por nombre (Factory)
    composite.py          # composición de métodos compatibles (Composite)
    naive.py              # finetune secuencial (baseline)
    ewc.py                # Elastic Weight Consolidation
    rehearsal.py          # memoria de repetición (buffer)
    as_snn.py             # Adaptive Synaptic Scaling
    sa_snn.py             # Sparse Selective Activation
    sca_snn.py            # Similarity-based Context-Aware
    colanet.py            # Columnar SNN

  prep/
    data_prep.py          # QC + normalización + splits (fusión de subvueltas)
    augment_offline.py    # balanceo por imágenes (genera imágenes reales por bins)
    encode_offline.py     # CSV → H5 (spikes)

tools/
  prep_offline.py         # CLI: SPLITS + (opcional) balanceo-img + (opcional) H5
  encode_offline.py       # CLI: CSV → H5 para un preset
  encode_tasks.py         # CLI: encode en bloque según presets
  run_continual.py        # CLI: entrenamiento continual por preset
  sim_drive.py            # Cliente del simulador (inferencia)
  README_sim.md           # Guía de integración con el simulador
```

---

## 4) Flujo de trabajo

### 4.1 Estructura RAW (múltiples runs por circuito)

Coloca tus recorridos del simulador Udacity así (ver `data/README.md` para más detalle):

```
data/raw/udacity/circuito1/vuelta1/{driving_log.csv, IMG/}
data/raw/udacity/circuito1/vuelta2/{driving_log.csv, IMG/}
data/raw/udacity/circuito2/vuelta1/{driving_log.csv, IMG/}
```


- Cada **circuito** puede tener **varias subvueltas** (`vuelta1`, `vuelta2`, …) que se **fusionan** en la preparación para formar un único conjunto por circuito.

### 4.2 Preparación de datos

Tienes dos opciones (elige una):

- **Opción A (clásica)** · `01_DATA_QC_PREP.ipynb`

Realiza QC (Quality Control) y normalización de rutas, fusión de subvueltas, estratificación por bins de steering y genera:

  - `data/processed/<run>/{canonical,train,val,test}.csv`
  - `data/processed/tasks.json (orden de tareas, p. ej. ["circuito1","circuito2"])`

- **Opción B (recomendada y autónoma)** · `01A_PREP_BALANCED.ipynb`

Ejecuta internamente los **SPLITS** (equivalentes a la opción A) y, además:

  - si activas `prep.balance_offline.mode: images` en el `presets.yaml`, genera `train_balanced.csv` con imágenes aumentadas reales por bins y `tasks_balanced.json`,
  - produce EDA rápida (histogramas/CSV por bins) por circuito para la memoria.

> No es necesario lanzar `01_DATA_QC_PREP.ipynb` si usas `01A_PREP_BALANCED.ipynb`.

### 4.3 Codificación a eventos (opcional offline)

- `02_ENCODE_OFFLINE.ipynb` convierte los CSV (imágenes) a **spikes offline** (`.h5`) conforme a tu preset (`encoder: rate/latency`, `T`, `gain`, tamaño, gris/color).
- Útil para comparativas controladas y depuración de E/S. Para la integración con simulador se usa **codificación online**.

### 4.4 Entrenamiento continual

- `03_TRAIN_CONTINUAL.ipynb` (o `tools/run_continual.py`) entrena una **secuencia de tareas** (p. ej., `circuito1 → circuito2`) con el método CL seleccionado en `configs/presets.yaml`.
- Métodos previstos/soportados:
  - naive (fine-tune secuencial),
  - ewc (Elastic Weight Consolidation),
  - rehearsal (memoria de repetición),
  - **SA-SNN, AS-SNN, SCA-SNN, CoLaNET** (bio-inspirados).

> Guarda métricas por tarea y olvido en `outputs/`.

### 4.5 Búsqueda de hiperparámetros (HPO)

- `03B_HPO_OPTUNA.ipynb` ejecuta Optuna con un **objetivo compuesto** que minimiza error final y penaliza el **olvido relativo**.
- Registra el estudio (`sqlite`) y un CSV de trials.

### 4.6 Resultados y figuras

- `04_RESULTS.ipynb` agrega resultados (MAE/MSE por tarea, olvido absoluto/relativo, boxplots por seed, etc.) y genera figuras/tablas para la memoria.

### 4.7 Integración con el simulador

- `99_SIM_INTEGRATION.ipynb` guía la inferencia en ciclo cerrado usando **codificación online** y el cliente `tools/sim_drive.py`.
Comprueba latencia/estabilidad y alinea parámetros con el preset (T, gain, tamaño, gris/color).

---

## 5) Presets y configuración

Archivo: `configs/presets.yaml`. Ejemplo mínimo (esquema actual):

```yaml
# =====================
# PRESETS TFM — SNN/CL
# =====================

# ---- Bloque base (anclas) ----
defaults: &defaults
  model: &defaults_model
    name: pilotnet_snn
    img_w: 200
    img_h: 66
    to_gray: true

  data: &defaults_data
    # Codificación
    encoder: rate              # rate | latency | raw | image
    T: 10
    gain: 0.5
    use_offline_spikes: true   # H5 offline
    encode_runtime: false      # solo si use_offline_spikes=false

    # Reproductibilidad / rendimiento
    seed: 42
    num_workers: 8
    prefetch_factor: 2
    pin_memory: true
    persistent_workers: true

    # Aumentos (ligeros por defecto)
    aug_train: &aug_light
      prob_hflip: 0.5
      brightness: [0.9, 1.1]
      contrast:   [0.9, 1.1]
      saturation: [0.9, 1.1]
      hue:        [-0.03, 0.03]
      gamma: null
      noise_std: 0.0

    # Balanceo online (desactivado por defecto)
    balance_online: false
    balance_bins: 50
    balance_smooth_eps: 0.001

  # División/prepare (offline)
  prep: &defaults_prep
    runs: ["circuito1", "circuito2"]  # o [] si autodetectas
    merge_subruns: true
    use_left_right: true
    steer_shift: 0.2
    bins: 50
    train: 0.70
    val: 0.15
    seed: 42
    # Balanceo offline de imágenes (si tu pipeline lo usa)
    balance_offline:
      mode: images            # images | none
      target_per_bin: auto
      cap_per_bin: 12000
      aug:
        prob_hflip: 0.0       # ← desactivar en offline, solo aumentos fotométricos
        brightness: [0.8, 1.2]
        contrast:   [0.8, 1.2]
        saturation: [0.8, 1.2]
        hue:        [-0.1, 0.1]
        gamma: null
        noise_std: 0.0
    tasks_file_name: tasks.json
    tasks_balanced_file_name: tasks_balanced.json
    use_balanced_tasks: true
    encode_h5: false   # (opcional). Se puede setear true en algún preset.

  optim: &defaults_optim
    amp: true
    lr: 0.001
    epochs: 2
    es_patience: null
    es_min_delta: null
    batch_size: 64

  # >>> EWC por defecto en TODOS los presets (salvo que se sobrescriba) <<<
  continual: &defaults_continual
    method: ewc
    params:
      lam: 1.0e9
      fisher_batches: 1000

  naming: &defaults_naming
    tag: ""


# ---- Presets ----

fast:
  <<: *defaults
  data:
    <<: *defaults_data
    T: 10
  optim:
    <<: *defaults_optim
    epochs: 2
    batch_size: 64
  continual:
    <<: *defaults_continual
    # EWC heredado (lam/fisher ya arriba)
  prep:
    <<: *defaults_prep
    # p.ej. bins: 40   # (opcional, si quisieras menos bins)

std:
  <<: *defaults
  data:
    <<: *defaults_data
    T: 16
  optim:
    <<: *defaults_optim
    lr: 5.0e-4
    epochs: 8
    batch_size: 56
    es_patience: 3
    es_min_delta: 1.0e-4
  continual:
    <<: *defaults_continual
    params:
      lam: 1.0e9
      fisher_batches: 1200
  prep:
    <<: *defaults_prep

accurate:
  <<: *defaults
  data:
    <<: *defaults_data
    T: 30
  optim:
    <<: *defaults_optim
    lr: 7.5e-4
    epochs: 20
    batch_size: 16
  continual:
    method: rehearsal+ewc
    params:
      buffer_size: 3000
      replay_ratio: 0.2
      lam: 1.0e9
      fisher_batches: 1500
  prep:
    <<: *defaults_prep
```

**Notas**

- `use_offline_spikes` y `encode_runtime` no deben estar a true simultáneamente.
- `01A_PREP_BALANCED.ipynb` respeta `prep.runs` si se definen; si no, **autodetecta** circuitos con `driving_log.csv` (en cualquier subcarpeta).

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

- **Prep completa (splits + balanceo por imágenes + H5 opcional)**  
  ```bash
  python tools/prep_offline.py --preset fast --config configs/presets.yaml --encode
  ```

- **Entrenamiento continual con preset**  
  ```bash
  python tools/run_continual.py --preset fast --config configs/presets.yaml --tag prueba1
  ```

- **Codificación H5 en bloque según presets**  
  ```bash
  python tools/encode_tasks.py --config configs/presets.yaml
  ```

- **Inferencia en simulador** (tras entrenar y exportar `model_best.pt`)  
  ```bash
  python tools/sim_drive.py --host 127.0.0.1 --port 4567 --model-path <ruta_a_tu_modelo.pt> \
    --model-name pilotnet_snn --img-w 200 --img-h 66 --to-gray \
    --encoder rate --T 10 --gain 0.5
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

- **naive**, **ewc**, **rehearsal**, **rehearsal+ewc**, **sa-snn**, **as-snn**, **sca-snn**, **colanet**.

---
