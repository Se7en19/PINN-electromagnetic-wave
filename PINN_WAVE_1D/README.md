# PINN_WAVE_1D

Physics-Informed Neural Network (PINN) para reconstruir la propagacion 1D del campo electrico `Ez` a partir de datos FDTD y la ecuacion de onda.

## Estructura

- `data/`: datos de entrada FDTD.
- `src/`: modelo, perdidas, entrenamiento, evaluacion y graficas.
- `results/`: artefactos generados durante la ejecucion.
- `main.py`: flujo principal de entrenamiento, validacion y guardado del modelo.

## Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecucion

Desde la carpeta del proyecto:

```bash
python main.py
```

El script:

- carga `data/Ez_FDTD.txt`,
- prepara puntos de entrenamiento y colocacion,
- entrena la PINN con Adam y L-BFGS,
- evalua cortes temporales y el mapa espacio-tiempo,
- guarda el modelo en `results/models/`.
