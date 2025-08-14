# Carpeta `data/` — organización de datos para el TFM (SNN + CL)

Esta carpeta **no** versiona datos ni artefactos pesados por defecto (lo indica `.gitignore`).  
Aquí irán los **datos RAW** del simulador Udacity y los **derivados** generados por los notebooks.

## Estructura esperada

```
data/
  raw/
    udacity/
      circuito1/
        driving_log.csv
        IMG/
          <imágenes .jpg>
      circuito2/
        driving_log.csv
        IMG/
          <imágenes .jpg>
  processed/
    circuito1/
      canonical.csv
      train.csv
      val.csv
      test.csv
      # (opcional) h5: p. ej. train_rate_T20_gain0.5.h5
    circuito2/
      canonical.csv
      train.csv
      val.csv
      test.csv
      # (opcional) h5…
  # marcadores para que Git suba las carpetas (vacías):
  .gitkeep
```

> Si tus recorridos tienen otro nombre (p. ej., `track1/track2`), puedes:
> - Renombrar las carpetas a `circuito1/circuito2`, o
> - Editar la variable `RUNS` en `notebooks/01_DATA_QC_PREP.ipynb`.

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

## Flujo de preparación

1. **Copia** tus datos RAW del simulador a:
   ```
   data/raw/udacity/circuito1/{driving_log.csv, IMG/}
   data/raw/udacity/circuito2/{driving_log.csv, IMG/}
   ```
2. Ejecuta `notebooks/01_DATA_QC_PREP.ipynb`:
   - Hace **QC** básico y normalización de rutas.
   - Crea **splits estratificados** por bins de `steering` → `train/val/test.csv`.
   - Genera `data/processed/tasks.json` con el orden de tareas (p. ej., `circuito1 → circuito2`).
3. (Opcional) `notebooks/02_ENCODE_OFFLINE.ipynb`:
   - Convierte imágenes a **spikes offline** (`.h5`) con tus parámetros (`T`, `gain`), útil para depurar y medir E/S.
4. Entrenamiento/Evaluación:
   - `notebooks/03_TRAIN_EVAL.ipynb` (o `tools/run_fast_ewc.py` si lo usas) consumen `processed/*` y `tasks.json`.

---

## Parámetros clave y decisiones

- **Codificación a impulsos**: por defecto *on-the-fly* (rate/latency); opción *offline* en HDF5.
- **Preprocesado**: gris, `160×80`, recorte opcional (editable en código).
- **Estratificación**: por bins de `steering` para evitar sesgo a “recta”.
- **Reproducibilidad**: semillas fijadas en `src/utils.py`.

---

## Problemas frecuentes

- **Rutas de Windows** con `\` → el notebook 01 las corrige a `IMG/...`.
- **Imágenes faltantes** respecto a `driving_log.csv` → se filtran (revisa `canonical.csv`).
- **Falta de GPU/CUDA** → el pipeline funciona en CPU, pero será más lento.
- **Memoria**: si te quedas sin VRAM, usa el preset `fast`, baja `batch_size` o `T`.

---

## Buenas prácticas

- No subas datos ni HDF5 al repo (ya está en `.gitignore`).
- Incluye en la memoria (o en `outputs/`) figuras/tablas que se generan con `04_RESULTS.ipynb`.
- Para nuevos recorridos, repite el paso **01** (se generan nuevos splits y se actualiza `tasks.json`).
