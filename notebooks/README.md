# Notebooks — guía de uso del pipeline

Este directorio reúne los cuadernos del TFM para **preparar datos**, **validar el pipeline datos→modelo**, **entrenar aprendizaje continuo**, **optimizar hiperparámetros**, **agregar resultados** y **probar la integración con el simulador**.

> Antes de empezar, revisa la organización de datos en [`data/README.md`](../data/README.md).

---

## Visión general

| Cuaderno | Para qué sirve | Entradas clave | Salidas clave | ¿Cuándo usarlo? |
|---|---|---|---|---|
| **01_DATA_QC_PREP.ipynb** | **QC** (control de calidad) y **splits estratificados** por *steering* | `data/raw/udacity/<circuito>/<vuelta>/{driving_log.csv, IMG/}` | `canonical.csv`, `{train,val,test}.csv`, `tasks.json`, `prep_manifest.json` | Inspección rápida y generación de splits limpios (sin balanceo por imágenes). |
| **01A_PREP_BALANCED.ipynb** | **Splits estratificados + balanceo offline** del `train` (opcional) | Igual que 01; puede leer `configs/presets.yaml` (`prep.*`) | Todo lo de 01 **+** `train_balanced.csv` (si activas balanceo) **+** `tasks_balanced.json` **+** EDA básica | Si quieres `train_balanced.csv` y/o fusionar **múltiples vueltas** por circuito. **Puede ejecutarse sola** (no requiere 01). |
| **02_DATA_SMOKE_BENCH.ipynb** | *Smokes* de **loader → forward** y **micro-bench** (*throughput*) | `presets.yaml`, `tasks.json`/`tasks_balanced.json`, H5 si procede | Mensajes de verificación, *it/s* de referencia | Antes de entrenar: detecta rutas rotas, H5 faltantes o desajustes de forma. |
| **02_ENCODE_OFFLINE.ipynb** | Generar H5 de **spikes offline** | `tasks.json` o `tasks_balanced.json`; preset (`encoder`, `T`, `gain`, tamaño, gris/color) | `train/val/test_{encoder}_T{T}_gain{…}_{gray|rgb}_{W}x{H}.h5` por run | **Necesario** si en el preset usas `data.use_offline_spikes: true`. |
| **03_TRAIN_CONTINUAL.ipynb** | Entrenamiento **continual** con el método del preset | `tasks*.json` y `presets.yaml` | Carpeta en `outputs/…` con métricas por tarea y resumen continual | Para correr comparativas base o por preset. |
| **03B_HPO_OPTUNA.ipynb** | **Búsqueda de hiperparámetros** (objetivo penaliza olvido) | `presets.yaml`, mismo *loader factory* que 03 | Estudio de Optuna, mejores *trials*, *logs* y resultados en `outputs/` | Cuando ya validaste el pipeline y quieres afinar métodos. |
| **04_RESULTS.ipynb** | **Agregación de resultados** y figuras | Carpeta(s) en `outputs/…` | Tablas/gráficos listos para la memoria | Tras entrenamientos/HPO. |
| **99_SIM_INTEGRATION.ipynb** | Prueba de **integración con el simulador** (ciclo cerrado) | Modelo entrenado, parámetros de codificación online | Logs/figuras cualitativas | Validar inferencia en tiempo real con codificación **runtime**. |

---

## Estructura de datos (recordatorio)

Se admiten **múltiples vueltas** por circuito:
```
data/
  raw/
    udacity/
      circuito1/
        vuelta1/{driving_log.csv, IMG/}
        vuelta2/{driving_log.csv, IMG/}
      circuito2/
        vuelta1/{driving_log.csv, IMG/}
        ...
  processed/
    circuito1/{canonical.csv, train.csv, val.csv, test.csv, [train_balanced.csv], eda/...}
    circuito2/{...}
    tasks.json
    [tasks_balanced.json]
    prep_manifest.json
```


- **`tasks.json`**: orden de tareas + rutas a `train/val/test.csv`.
- **`tasks_balanced.json`**: igual, pero `train` apunta a `train_balanced.csv`.
- **`prep_manifest.json`**: trazabilidad de la preparación (parámetros y rutas generadas).

---

## Flujo recomendado (recetas)

### A) Vía **offline** (H5 de spikes) — comparativas controladas
1. `01A_PREP_BALANCED.ipynb` → generar splits y (si quieres) `train_balanced.csv` + `tasks_balanced.json`.  
2. `02_ENCODE_OFFLINE.ipynb` → crear H5 en `data/processed/<run>/`.  
3. `02_DATA_SMOKE_BENCH.ipynb` → validar que existen H5 y que el *forward* funciona.  
4. `03_TRAIN_CONTINUAL.ipynb` → entrenar según `continual.method` del preset.  
5. `04_RESULTS.ipynb` → figuras/tablas.

> En tu preset: `data.use_offline_spikes: true` y `data.encode_runtime: false`.

### B) Vía **runtime** (codificación en GPU) — integración / rapidez
1. `01A_PREP_BALANCED.ipynb` (o `01_DATA_QC_PREP.ipynb` si no necesitas balanceo).  
2. Ajustar preset: `data.use_offline_spikes: false` y `data.encode_runtime: true`.  
3. `02_DATA_SMOKE_BENCH.ipynb` → verificar pipeline con encode en runtime.  
4. `03_TRAIN_CONTINUAL.ipynb` → entrenar.  
5. `99_SIM_INTEGRATION.ipynb` → probar en el simulador (ciclo cerrado).

---

## ¿Qué es “QC” y en qué se diferencian 01 y 01A?

- **QC (Quality Check / Control de calidad)**: lectura robusta de `driving_log.csv`, normalización de rutas (corrige separadores y recorta a `IMG/...`), filtrado de filas sin imagen válida y **estratificación** del *split* por *steering* (bins).

- **01_DATA_QC_PREP**: QC + splits **sin** balanceo por imágenes.  
- **01A_PREP_BALANCED**: hace lo anterior y añade:
  - Fusión de **múltiples vueltas** por circuito en los CSV del circuito.
  - **Balanceo offline por imágenes** (genera material cuando faltan muestras por bin).
  - EDA rápida (histogramas y conteos por bin).

> **Nota**: *01A puede ejecutarse sola*; usa 01 si solo quieres revisar y generar splits básicos.

---

## Entradas/salidas por cuaderno

- **01 / 01A**
  - **In**: `raw/udacity/*/vuelta*/{driving_log.csv, IMG/}`
  - **Out**: `processed/<run>/{canonical,train,val,test}.csv`, `tasks.json` y (opcional en 01A) `train_balanced.csv`, `tasks_balanced.json`, EDA.

- **02_ENCODE_OFFLINE**
  - **In**: `tasks.json` o `tasks_balanced.json`, preset (`encoder`, `T`, `gain`, `img_w/img_h`, `to_gray`).
  - **Out**: `processed/<run>/{split}_{encoder}_T{T}_gain{…}_{gray|rgb}_{W}x{H}.h5`.

- **02_DATA_SMOKE_BENCH**
  - **In**: preset, `tasks*.json` y (si `use_offline_spikes:true`) H5.
  - **Out**: verificación de disponibilidad de datos y *throughput* de referencia.

- **03_TRAIN_CONTINUAL**
  - **In**: preset (`continual.method` y `params`), `tasks*.json`, datos (H5 o CSV).
  - **Out**: carpeta en `outputs/` con manifiestos y métricas por tarea y resumen continual.

- **03B_HPO_OPTUNA**
  - **In**: preset base; define búsqueda y objetivo compuesto (error final + penalización de olvido relativo).
  - **Out**: estudio/ficheros de Optuna y métricas agregadas.

- **04_RESULTS**
  - **In**: una o más carpetas `outputs/...`
  - **Out**: tablas y figuras para la memoria.

- **99_SIM_INTEGRATION**
  - **In**: modelo entrenado y parámetros de codificación **online**.
  - **Out**: validación cualitativa de lazo cerrado (simulador).

---

## Verificaciones y guardarraíles

- **Selección de tasks**: si el preset indica `prep.use_balanced_tasks: true`, se elige `tasks_balanced.json` **si existe**; de lo contrario, se usa `tasks.json`.  
  - En modo “balanceado”, se **exige** que `train` apunte a `train_balanced.csv`.

- **Modo H5 offline**: si `data.use_offline_spikes: true`, se comprueba que existen los H5 esperados por **split** con *naming* consistente (encoder/T/gain/color/size).

- **Modo runtime**: se valida que los CSV existen y que una muestra inicial tiene **rutas de imagen válidas** (si esperas material en `aug/` y no está, se avisa).

---

## Problemas frecuentes (y solución)

- **Faltan H5** en modo offline → generar con `02_ENCODE_OFFLINE.ipynb` **o** desactivar H5 (preset) y activar `encode_runtime`.
- **Rutas de imagen no válidas** en CSV → re-ejecutar `01A_PREP_BALANCED` para normalizar y regenerar material balanceado si procede.
- **Desajustes de forma** (se espera 5D y llega 4D) → revisar `ENCODER/T/GAIN` en el preset y la permuta/codificación en `02_DATA_SMOKE_BENCH`.
- **Bloqueos del loader** → reducir `num_workers`/`prefetch_factor` y/o desactivar `persistent_workers`.
- **VRAM** insuficiente → usar preset `fast`, bajar `batch_size` y/o `T`, y mantener `optim.amp: true`.

---

## Convenciones y reproducibilidad

- **Presets como fuente de verdad**: define en `configs/presets.yaml` el modelo base, codificación temporal, *augment*, *loader* y método continual.
- **Semillas**: el preset fija `data.seed`; los cuadernos propagan ese valor para splits, codificación y entrenamiento.
- **Nomenclatura coherente**: H5, carpetas de `outputs/` y figuras siguen *naming* estable con `encoder/T/gain/color/size` y el `preset_name`.
