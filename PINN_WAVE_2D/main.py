import sys
sys.path.append('/kaggle/input')

import numpy as np
import tensorflow as tf

from src.data import (
    cargar_datos_fdtd,
    preparar_datos_entrenamiento,
    preparar_puntos_colocacion,
    preparar_constantes_fisicas,
)
from src.model import PINN_WAVE
from src.train import entrenar_adam, entrenar_lbfgs
from src.evaluate import visualizar_snapshot, validar_modelo_final
from src.plotting import generar_mapas_ez, plot_historial_perdidas


np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)


def main():
    fdtd_data = cargar_datos_fdtd('/content/Resultados_Simulacion.mat')
    training_data = preparar_datos_entrenamiento(fdtd_data)
    colocacion_data = preparar_puntos_colocacion(training_data)
    phys_data = preparar_constantes_fisicas(training_data, colocacion_data)

    model = PINN_WAVE()
    model.summary()

    x = training_data['x']
    y = training_data['y']
    t = training_data['t']
    Ez = training_data['Ez']
    Q = training_data['Q']
    escalar_x = training_data['escalar_x']
    escalar_y = training_data['escalar_y']
    escalar_t = training_data['escalar_t']
    descalar_u = training_data['descalar_u']

    def callback_snapshot(epoch, model_cb):
        idx_test = Q // 2
        rell_err = visualizar_snapshot(
            model_cb, x, y, t, Ez, idx_test,
            escalar_x, escalar_y, escalar_t, descalar_u,
            titulo=f'Adam epoch {epoch}',
            guardar=f'/content/snap_epoch_{epoch}.png'
        )
        print(f"Snapshot t={t[idx_test]:.2e}s | Err={rell_err:.3f}")

    adam_results = entrenar_adam(
        model=model,
        training_data=training_data,
        colocacion_data=colocacion_data,
        phys_data=phys_data,
        callback_snapshot=callback_snapshot,
    )

    lbfgs_results = entrenar_lbfgs(
        model=model,
        training_data=training_data,
        colocacion_data=colocacion_data,
        phys_data=phys_data,
        adam_results=adam_results,
    )

    validar_modelo_final(model, training_data)

    generar_mapas_ez(model, training_data)
    plot_historial_perdidas(adam_results, lbfgs_results)

    path_model = '/content/PINN_2D_WAVE_s42.h5'
    model.save(path_model)
    print(f"\nModelo guardado en: {path_model}")


if __name__ == '__main__':
    main()
