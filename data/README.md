# Carpeta `data/` — organización de datos para el TFM (SNN + CL)

Esta carpeta **no** versiona datos ni artefactos pesados por defecto (lo indica `.gitignore`).  
Aquí irán los **datos RAW** del simulador Udacity y los **derivados** que generan los notebooks/herramientas.

## Estructura esperada

```
data/
  raw/
    udacity/
      circuito1/
        vuelta1/
          driving_log.csv
          IMG/
            <imágenes .jpg>
        vuelta2/
          driving_log.csv
          IMG/
            <imágenes .jpg>
      circuito2/
        vuelta1/
          driving_log.csv
          IMG/
            <imágenes .jpg>
        ...
  processed/
    circuito1/
      canonical.csv         # dataset canónico (pre-expansión L/R)
      train.csv             # split de entrenamiento (post-fusión subvueltas)
      val.csv
      test.csv
      train_balanced.csv    # (opcional) balanceo por imágenes reales (por bins)
      # (opcional) HDF5: p. ej. train_rate_T10_gain0.5_gray_200x66.h5
    circuito2/
      canonical.csv
      train.csv
      val.csv
      test.csv
      train_balanced.csv    # si activas balanceo offline
  processed/tasks.json
  processed/tasks_balanced.json  # si activas balanceo offline por imágenes
  .gitkeep
```

## ¿Qué hace cada notebook de preparación?

- `01_DATA_QC_PREP.ipynb`
  - QC (Quality Control): normaliza rutas, corrige separadores (`\` → `/`), filtra imágenes inexistentes.
  - Fusión de subvueltas (`merge_subruns: true`): combina `vuelta1`, `vuelta2`, … en un único conjunto por circuito.
  - Expansión L/R (`use_left_right: true`): usa cámaras izquierda/derecha aplicando una corrección de `steer_shift`.
  - Estratificación por bins de steering → `train/val/test.csv`.
  - Genera `processed/tasks.json` con el orden de tareas (p. ej., `["circuito1","circuito2"]`).
- `01A_PREP_BALANCED.ipynb`
  - Ejecuta internamente lo anterior (SPLITS).
  - Si activas `prep.balance_offline.mode: images`, genera train_balanced.csv creando imágenes aumentadas reales para rellenar bins deficitarios y escribe `tasks_balanced.json`.
  - Incluye EDA rápida por circuito: histogramas de steering y tabla CSV de cuentas por bin.

> No necesitas ejecutar `01_DATA_QC_PREP.ipynb` si usas `01A_PREP_BALANCED.ipynb`.


---

## Formato del `driving_log.csv`

Columnas estándar de Udacity (sin cabecera en origen):
```
center,left,right,steering,throttle,brake,speed
```

- **Rutas de imagen** (`center/left/right`): pueden venir absolutas o relativas.  
  El notebook **01** normaliza automáticamente a rutas tipo `IMG/xxxxx.jpg` y **filtra** las que no existan en disco.
- **steering**: float en `[-1, +1]` (ángulo normalizado).
- **throttle, brake, speed**: no se usan para el objetivo principal, pero se conservan.

---

## Parámetros clave (preset → `prep`)

- `runs`: lista de circuitos a preparar; si va vacío, se **autodetectan** los que contengan algún `driving_log.csv`.
- `merge_subruns`: true para fusionar `vuelta1`, `vuelta2`, …
- `use_left_right`: true para expandir cámaras L/R con corrección `steer_shift`.
- `bins`: nº de bins de steering para estratificación/balanceo.
- `balance_offline.mode`: `"images"` para generar `train_balanced.csv` con **aumentación real** (no duplicación de filas).
- `tasks_file_name` / `tasks_balanced_file_name`: nombres de los JSON de tareas.

---

## Problemas frecuentes

- **Rutas de Windows** con `\` → el notebook 01 las corrige a `IMG/...`.
- **Imágenes faltantes** respecto a `driving_log.csv` → se filtran (revisa `canonical.csv`).
- **Distribuciones muy sesgadas a recta** → usa el balanceo por imágenes (`prep.balance_offline.mode: images`).
- **Falta de GPU/CUDA** → el pipeline funciona en CPU, pero será más lento.
- **Memoria**: si te quedas sin VRAM, usa el preset `fast`, baja `batch_size` o `T`.
