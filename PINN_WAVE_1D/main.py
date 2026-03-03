from pathlib import Path
import numpy as np
import tensorflow as tf
from src.data import preparar_datos_entrenamiento, preparar_puntos_colocacion
from src.evaluate import evaluar_cortes, generar_mapa_espacio_tiempo, predecir_corte
from src.model import PINN_WAVE
from src.plotting import (
    plot_corte_adam,
    plot_cortes_validacion,
    plot_historial_perdidas,
    plot_mapa_espacio_tiempo,
)
from src.train import entrenar_adam, entrenar_lbfgs


def main():
    # Execute the complete training, evaluation and saving process
    
    # Find the path of the FDTD data file 
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "data" / "Ez_FDTD.txt"

    # Prepare training data from FDTD field
    training_data = preparar_datos_entrenamiento(path, n_sparse=150, source_x=1.5)

    # Extract the training data
    espacio = training_data["espacio"]
    tiempo = training_data["tiempo"]
    EzenXyT = training_data["EzenXyT"]
    delta_x = training_data["delta_x"]
    delta_t = training_data["delta_t"]
    renglon = training_data["renglon"]
    columna = training_data["columna"]

    # Print the dimensions of the spatial and temporal coordinates and the physical field 
    print("Dimension del espacio: ", espacio.shape)
    print("Dimension del tiempo:  ", tiempo.shape)
    print("Dimension de Ez:       ", EzenXyT.shape)
    print(
        f"Nodo de la fuente: {training_data['nodo_fuente']} | "
        f"x = {espacio[training_data['nodo_fuente']]:.4f} m"
    )
    print(f"\nTotal puntos de entrenamiento: {training_data['X0'].shape[0]}")
    print(
        f"  ({100 * training_data['X0'].shape[0] / (renglon * columna):.2f}% del dominio total)"
    )

    # build points of colocation using Latin Hypercube Sampling 
    colocacion = preparar_puntos_colocacion(
        espacio,
        delta_x,
        training_data["lb_x"],
        training_data["ub_x"],
        n_colocation=50000,
        source_x=1.5,
    )

    print(f"\nPuntos de colocación física: {colocacion['X'].shape[0]}")


    # PHYSICAL PARAMETERS 
    e0 = 8.8541e-12
    m0 = 4 * np.pi * 1e-7
    c0 = tf.cast(1.0 / np.sqrt(e0 * m0), dtype=tf.float32)

    k_x = tf.constant(
        2.0 / (training_data["ub_x"] - training_data["lb_x"]),
        dtype=tf.float32,
    )
    k_t = tf.constant(
        2.0 / (training_data["ub_t"] - training_data["lb_t"]),
        dtype=tf.float32,
    )

    lambda_sq_check = (c0.numpy() * k_x.numpy() / k_t.numpy()) ** 2
    print(f"\nc0        = {c0.numpy():.4e} m/s")
    print(f"k_x       = {k_x.numpy():.4e} m⁻¹")
    print(f"k_t       = {k_t.numpy():.4e} s⁻¹")
    print(f"lambda_sq = {lambda_sq_check:.4f}")

    # MODEL
    model = PINN_WAVE()
    model.summary()


    # Plot a figure of the prediction of the physical field at a specific time every 10k epochs 
    def callback_plot_adam(epoch, model):
        if epoch % 10000 != 0:
            return

        idx_test = 50
        t_test_val = tiempo[idx_test]
        u_test_real = EzenXyT[:, idx_test]
        _, _, u_pred = predecir_corte(
            model,
            espacio,
            t_test_val,
            training_data["lb_x"],
            training_data["ub_x"],
            training_data["lb_t"],
            training_data["ub_t"],
            training_data["u_mean"],
            training_data["u_std"],
        )
        plot_corte_adam(
            espacio,
            u_test_real,
            u_pred,
            training_data["x_sparse"],
            training_data["t_sparse"],
            training_data["u_sparse"],
            t_test_val,
            delta_t,
            epoch,
            source_x=1.5,
        )

    # Traning the model using Adam optimizer
    adam_results = entrenar_adam(
        model,
        training_data["X0"],
        training_data["T0"],
        training_data["U0_norm"],
        colocacion["X"],
        colocacion["T"],
        k_x,
        k_t,
        c0,
        epochs_adam=35000,
        lr_adam=1e-3,
        w_data=100.0,
        w_phys=1.0,
        epoch_inicio_fisica=20000,
        ramp_fisica=10000.0,
        print_every=500,
        callback_epoch=callback_plot_adam,
    )
    # Refine the model using L-BFGS optimizer
    lbfgs_results = entrenar_lbfgs(
        model,
        training_data["X0"],
        training_data["T0"],
        training_data["U0_norm"],
        colocacion["X"],
        colocacion["T"],
        k_x,
        k_t,
        c0,
        w_data=100.0,
        w_phys=1.0,
        maxiter=5000,
        maxfun=50000,
        ftol=1e-12,
        gtol=1e-8,
        maxls=50,
        print_every=100,
    )

    ######################################################
    # PLOTTING
    ######################################################

    print("\nGenerando graficas de validacion final...")
    evaluation_results = evaluar_cortes(
        model,
        espacio,
        tiempo,
        EzenXyT,
        indices_tiempo=[10, 25, 40, 55, 70, 85],
        lb_x=training_data["lb_x"],
        ub_x=training_data["ub_x"],
        lb_t=training_data["lb_t"],
        ub_t=training_data["ub_t"],
        u_mean=training_data["u_mean"],
        u_std=training_data["u_std"],
    )
    plot_cortes_validacion(evaluation_results["resultados"], espacio, source_x=1.5)
    print(
        f"\nError relativo promedio: "
        f"{evaluation_results['error_relativo_promedio']:.4f}"
    )

    print("\nGenerando mapa espacio-tiempo...")
    field_map = generar_mapa_espacio_tiempo(
        model,
        espacio,
        tiempo,
        EzenXyT,
        training_data["lb_x"],
        training_data["ub_x"],
        training_data["lb_t"],
        training_data["ub_t"],
        training_data["u_mean"],
        training_data["u_std"],
    )
    plot_mapa_espacio_tiempo(
        tiempo,
        espacio,
        EzenXyT,
        field_map["Ez_pred_map"],
        field_map["err_map"],
        training_data["t_sparse"],
        training_data["x_sparse"],
        n_sparse=150,
    )

    plot_historial_perdidas(
        adam_results["loss_history"],
        adam_results["data_history"],
        adam_results["phys_history"],
        lbfgs_results["lbfgs_losses"],
        epochs_adam=35000,
        epoch_inicio_fisica=20000,
    )

    path_model = base_dir / "results" / "models" / "PINN_WAVE_corrected.h5"
    path_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(path_model)
    print(f"\nModelo guardado en {path_model}")


if __name__ == "__main__":
    main()
