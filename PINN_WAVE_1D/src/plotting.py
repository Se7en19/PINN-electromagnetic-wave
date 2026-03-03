import matplotlib.pyplot as plt
import numpy as np


def plot_corte_adam(
    espacio,
    u_test_real,
    u_pred,
    x_sparse,
    t_sparse,
    u_sparse,
    t_test_val,
    delta_t,
    epoch,
    source_x=1.5,
):
    # Plot the prediction of the physical field at a specific time
    mask_t = np.abs(t_sparse.flatten() - t_test_val) < (2 * delta_t)

    plt.figure(figsize=(10, 4))
    plt.plot(espacio, u_test_real, "b-", lw=2, label="Real (FDTD)")
    plt.plot(espacio, u_pred, "r--", lw=1.5, label="PINN (Adam)")
    if mask_t.any():
        plt.scatter(
            x_sparse[mask_t],
            u_sparse[mask_t],
            c="lime",
            s=40,
            zorder=5,
            label="Datos dispersos",
        )
    plt.axvline(x=source_x, color="g", linestyle=":", alpha=0.6)
    plt.title(f"[Adam] Epoch {epoch} | t = {t_test_val:.2e} s")
    plt.xlabel("Espacio (m)")
    plt.ylabel("Ez (V/m)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cortes_validacion(resultados, espacio, source_x=1.5):
    # Plot the temporal cross-sections used in the final validation
    for resultado in resultados:
        plt.figure(figsize=(10, 4))
        plt.plot(espacio, resultado["u_test_real"], "b-", lw=2, label="Real (FDTD)")
        plt.plot(
            espacio,
            resultado["u_pred"],
            "r--",
            lw=1.5,
            label="PINN (Adam+L-BFGS)",
        )
        plt.axvline(
            x=source_x,
            color="g",
            linestyle=":",
            alpha=0.6,
            label="Fuente x=1.5m",
        )
        plt.title(
            f't = {resultado["t_test_val"]:.2e} s  |  '
            f'MSE = {resultado["mse_val"]:.3e}  |  '
            f'Err rel = {resultado["rel_err"]:.3f}'
        )
        plt.xlabel("Espacio (m)")
        plt.ylabel("Ez (V/m)")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_mapa_espacio_tiempo(
    tiempo,
    espacio,
    EzenXyT,
    Ez_pred_map,
    err_map,
    t_sparse,
    x_sparse,
    n_sparse,
):
    # Plot the physical field, the prediction and the absolute error map
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(
        EzenXyT,
        aspect="auto",
        extent=[tiempo.min(), tiempo.max(), espacio.max(), espacio.min()],
        cmap="RdBu_r",
    )
    axes[0].scatter(
        t_sparse,
        x_sparse,
        c="lime",
        s=8,
        alpha=0.7,
        label=f"{n_sparse} pts entrenamiento",
    )
    axes[0].set_title("FDTD (Real)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Espacio (m)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        Ez_pred_map,
        aspect="auto",
        extent=[tiempo.min(), tiempo.max(), espacio.max(), espacio.min()],
        cmap="RdBu_r",
        vmin=EzenXyT.min(),
        vmax=EzenXyT.max(),
    )
    axes[1].set_title("PINN (Adam + L-BFGS)")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Espacio (m)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err_map,
        aspect="auto",
        extent=[tiempo.min(), tiempo.max(), espacio.max(), espacio.min()],
        cmap="hot_r",
    )
    axes[2].set_title("Error Absoluto |FDTD - PINN|")
    axes[2].set_xlabel("Tiempo (s)")
    axes[2].set_ylabel("Espacio (m)")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f"PINN con Adam + L-BFGS | {n_sparse} puntos dispersos", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_historial_perdidas(
    loss_history,
    data_history,
    phys_history,
    lbfgs_losses,
    epochs_adam,
    epoch_inicio_fisica=20000,
):
    # Plot the evolution of the training losses
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(loss_history, "b-", lw=0.8, label="Adam")
    if lbfgs_losses:
        x_lbfgs = np.linspace(
            epochs_adam,
            epochs_adam + len(lbfgs_losses) * 100,
            len(lbfgs_losses),
        )
        axes[0].plot(x_lbfgs, lbfgs_losses, "r-", lw=0.8, label="L-BFGS")
    axes[0].axvline(
        x=epochs_adam,
        color="g",
        linestyle="--",
        alpha=0.7,
        label="Inicio L-BFGS",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Loss Total: Adam → L-BFGS")
    axes[0].set_xlabel("Epochs / Iteraciones")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data_history, "b-", lw=0.8, label="Datos (×100)")
    axes[1].plot(phys_history, "r-", lw=0.8, label="Física")
    axes[1].axvline(
        x=epoch_inicio_fisica,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="Inicio física",
    )
    axes[1].axvline(
        x=epochs_adam,
        color="g",
        linestyle="--",
        alpha=0.7,
        label="Inicio L-BFGS",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Loss Datos vs Física (Adam)")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
