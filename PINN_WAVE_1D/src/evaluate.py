import numpy as np
import tensorflow as tf

from src.data import descalar_u, escalar_t, escalar_x, to_tensor


def predecir_corte(model, espacio, t_test_val, lb_x, ub_x, lb_t, ub_t, u_mean, u_std):
    x_test_col = espacio.reshape(-1, 1)
    t_test_col = np.full_like(x_test_col, t_test_val)
    inputs_test = tf.concat(
        [
            to_tensor(escalar_x(x_test_col, lb_x, ub_x)),
            to_tensor(escalar_t(t_test_col, lb_t, ub_t)),
        ],
        axis=1,
    )
    u_pred = descalar_u(model(inputs_test, training=False).numpy(), u_mean, u_std)
    return x_test_col, t_test_col, u_pred


def evaluar_cortes(
    model,
    espacio,
    tiempo,
    EzenXyT,
    indices_tiempo,
    lb_x,
    ub_x,
    lb_t,
    ub_t,
    u_mean,
    u_std,
):
    resultados = []
    errores_relativos = []

    for idx_test in indices_tiempo:
        t_test_val = tiempo[idx_test]
        u_test_real = EzenXyT[:, idx_test]

        _, _, u_pred = predecir_corte(
            model,
            espacio,
            t_test_val,
            lb_x,
            ub_x,
            lb_t,
            ub_t,
            u_mean,
            u_std,
        )

        mse_val = np.mean((u_pred.flatten() - u_test_real) ** 2)
        rel_err = np.linalg.norm(u_pred.flatten() - u_test_real) / (
            np.linalg.norm(u_test_real) + 1e-12
        )
        errores_relativos.append(rel_err)

        resultados.append(
            {
                "idx_test": idx_test,
                "t_test_val": t_test_val,
                "u_test_real": u_test_real,
                "u_pred": u_pred,
                "mse_val": mse_val,
                "rel_err": rel_err,
            }
        )

    return {
        "resultados": resultados,
        "errores_relativos": errores_relativos,
        "error_relativo_promedio": np.mean(errores_relativos),
    }


def generar_mapa_espacio_tiempo(
    model,
    espacio,
    tiempo,
    EzenXyT,
    lb_x,
    ub_x,
    lb_t,
    ub_t,
    u_mean,
    u_std,
):
    # Evaluate the model over the spatial-temporal grid
    X_grid, T_grid = np.meshgrid(espacio, tiempo, indexing="ij")
    X_flat = X_grid.reshape(-1, 1)
    T_flat = T_grid.reshape(-1, 1)

    inputs_all = tf.concat(
        [
            to_tensor(escalar_x(X_flat, lb_x, ub_x)),
            to_tensor(escalar_t(T_flat, lb_t, ub_t)),
        ],
        axis=1,
    )
    Ez_pred_map = descalar_u(
        model(inputs_all, training=False).numpy(),
        u_mean,
        u_std,
    ).reshape(EzenXyT.shape[0], EzenXyT.shape[1])

    err_map = np.abs(EzenXyT - Ez_pred_map)

    return {
        "X_grid": X_grid,
        "T_grid": T_grid,
        "X_flat": X_flat,
        "T_flat": T_flat,
        "Ez_pred_map": Ez_pred_map,
        "err_map": err_map,
    }
