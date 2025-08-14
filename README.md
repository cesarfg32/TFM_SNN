# TFM — SNN + Aprendizaje Continuo en SNN (steering Udacity)

Proyecto para investigar **aprendizaje continuo (Continual Learning, CL)** en **redes de impulsos (Spiking Neural Networks, SNN)** aplicadas a **regresión de ángulo de dirección (steering)** en conducción simulada (Udacity).  
Stack: **PyTorch + snnTorch**, ejecución local en **Linux/WSL2 + CUDA** o CPU.

> 📁 **Datos**: ver guía completa en [`data/README.md`](data/README.md).


## 1) Requisitos

- **Python 3.12** (recomendado) y `pip`.
- Linux o **WSL2 (Ubuntu 24.04)** en Windows 11.
- GPU NVIDIA opcional (recomendado). Para CUDA 12.9 en RTX 4080 se usa la build `+cu129` de PyTorch.
- Compilación no requerida: todo el código está en Python.

> Si trabajas en WSL2, asegúrate de tener el driver NVIDIA actualizado en Windows y soporte CUDA en WSL (`nvidia-smi` dentro de WSL).


## 2) Instalación rápida

```bash
# Clona el repo
git clone https://github.com/TU_GITHUB/TFM_SNN.git
cd TFM_SNN

# Crea y activa el entorno del proyecto
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Instala PyTorch para tu GPU (ejemplo: CUDA 12.9)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Resto de dependencias
pip install -r requirements.txt

# Comprobación de entorno (versiones, CUDA, GPU, estructura)
python tools/check_env.py
```


## 3) Organización del proyecto

```
configs/
  presets.yaml           # perfiles de ejecución (fast / std / accurate)
data/
  raw/udacity/...        # datos originales (ver data/README.md)
  processed/...          # splits y derivados generados por los notebooks
notebooks/
  01_DATA_QC_PREP.ipynb  # QC + normalización rutas + splits (train/val/test) + tasks.json
  02_ENCODE_OFFLINE.ipynb# (opcional) codificación a spikes offline (HDF5)
  03_TRAIN_EVAL.ipynb    # entrenamiento supervised y continual (EWC baseline)
  04_RESULTS.ipynb       # agregación de resultados y figuras
outputs/
  ...                    # métricas, manifiestos y resultados por experimento
src/
  encoders.py            # codificadores rate/latency (on-the-fly)
  datasets.py            # dataloaders Udacity + HDF5 opcional
  models.py              # backbone SNNVisionRegressor (CNN + LIF)
  training.py            # bucles de entrenamiento (supervised y continual)
  metrics.py             # MAE/MSE y BWT (olvido)
  utils.py               # seeds, presets, dataloaders auxiliares
  methods/
    ewc.py               # Elastic Weight Consolidation (baseline)
    sa_snn.py            # stub (activación selectiva escasa)
    as_snn.py            # stub (escalado sináptico adaptativo)
    sca_snn.py           # stub (context-aware por similaridad)
    colanet.py           # stub (arquitectura columnar)
tools/
  check_env.py           # verifica entorno, CUDA y estructura
  smoke_train.py         # entrenamiento sintético rápido (sin datos)
```


## 4) Flujo de trabajo

### 4.1 Preparar datos
Coloca tus recorridos del simulador Udacity así (ver detalles en `data/README.md`):
```
data/raw/udacity/circuito1/driving_log.csv  +  data/raw/udacity/circuito1/IMG/
data/raw/udacity/circuito2/driving_log.csv  +  data/raw/udacity/circuito2/IMG/
```

Ejecuta el notebook **01_DATA_QC_PREP.ipynb** (selecciona el kernel de `.venv`):
- Normaliza rutas a `IMG/...`, filtra imágenes inexistentes y guarda `canonical.csv`.
- Genera **splits estratificados** por bins de `steering`: `train/val/test.csv`.
- Crea `data/processed/tasks.json` con el orden de tareas (p. ej., `["circuito1","circuito2"]`).

> *(Opcional)* Ejecuta **02_ENCODE_OFFLINE.ipynb** para crear HDF5 con spikes si quieres depurar E/S o medir diferencias offline vs on‑the‑fly.


### 4.2 Entrenamiento y evaluación
Abre **03_TRAIN_EVAL.ipynb** y ejecuta:
- **Supervised** en `circuito1` con preset `fast` (rápido para comprobar que todo funciona).
- **Continual** `circuito1 → circuito2` con **EWC** (baseline).  
Se guardan métricas en `outputs/` (ver estructura abajo).

Luego abre **04_RESULTS.ipynb** para ver:
- Tabla con MAE/MSE de validación/tiempo final.
- Resumen de continual (`continual_results.json`) y gráfica simple.


## 5) Presets y configuración

Archivo: `configs/presets.yaml`

- `fast`: pruebas rápidas (pocas épocas, T reducido).  
- `std`: equilibrio tiempo/calidad.  
- `accurate`: más épocas y ventana temporal (T) mayor para resultados finales.

Cada preset define: `epochs`, `batch_size`, `T` (pasos temporales), `gain` (en rate), `encoder` (rate/latency), `lr`, `amp` (mixed precision).


## 6) Estructura de resultados (`outputs/`)

Por cada ejecución se crea una carpeta con:
- `metrics.json`: histórico de MAE/MSE (train/val) por época.
- `manifest.json`: metadatos mínimos (modo, preset, device, tiempos, etc.).
- En continual: `continual_results.json` con métricas por tarea y degradaciones tras nuevas tareas.

Ejemplo de carpeta:
```
outputs/
  supervised_fast_ewc0/
    metrics.json
    manifest.json
  continual_fast_ewc/
    task_1_circuito1/
      metrics.json
      manifest.json
    task_2_circuito2/
      metrics.json
      manifest.json
    continual_results.json
```


## 7) Scripts útiles (sin notebooks)

- **Comprobación de entorno**
  ```bash
  python tools/check_env.py
  ```

- **Smoke test de entrenamiento sintético** (no requiere datos):
  ```bash
  python tools/smoke_train.py --steps 50 --T 10 --batch 8 --amp
  ```
  Si ves `Dispositivo: cuda` y la `loss` baja, tu stack está OK.


## 8) Resolución de problemas frecuentes

- **`CUDA disponible: False`**  
  Revisa driver en Windows, soporte CUDA en WSL, y reinstala PyTorch con el índice de tu versión CUDA (`cu129` en RTX 4080).

- **`ModuleNotFoundError: No module named 'torch'`**  
  Asegúrate de tener activado el entorno `.venv` y de haber ejecutado la instalación de PyTorch y `requirements.txt`.

- **`FileNotFoundError: data/processed/tasks.json`**  
  Ejecuta antes `notebooks/01_DATA_QC_PREP.ipynb` para generar los splits y `tasks.json`.

- **OOM (falta VRAM)**  
  Usa preset `fast`, baja `batch_size` o `T`, y mantén `amp: true`.

- **Kernel equivocado en notebooks**  
  Selecciona el intérprete de `.venv` en VS Code (Ctrl+Shift+P → *Python: Select Interpreter*).


## 9) Estado de métodos CL

- **EWC** (baseline) implementado en `src/methods/ewc.py` e integrado en el notebook 03.  
- **SA‑SNN / AS‑SNN / SCA‑SNN / CoLaNET** incluidos como *stubs* en `src/methods/` para su desarrollo incremental y comparación bajo el mismo pipeline.


---

¿Dudas? Abre un issue o comenta sobre qué método quieres implementar primero (SA‑SNN o AS‑SNN son buenas opciones para continuar).
