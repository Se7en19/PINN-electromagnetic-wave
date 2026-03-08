import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def generar_mapas_ez(model, training_data):
    print("\nGenerando mapas Ez(x,y)...")

    x = training_data['x']
    y = training_data['y']
    t = training_data['t']
    Ez = training_data['Ez']
    Mx = training_data['Mx']
    My = training_data['My']
    Q = training_data['Q']
    nodo_fuente = training_data['nodo_fuente']
    n_sparse = training_data['n_sparse']
    escalar_x = training_data['escalar_x']
    escalar_y = training_data['escalar_y']
    escalar_t = training_data['escalar_t']
    descalar_u = training_data['descalar_u']
    to_tensor = training_data['to_tensor']

    idx_mapas = [0, Q // 4, Q // 2, 3 * Q // 4, Q - 1]
    fig, axes = plt.subplots(2, len(idx_mapas), figsize=(20, 8))

    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    Xf_all = Xg.reshape(-1, 1)
    Yf_all = Yg.reshape(-1, 1)

    for col, idx_t in enumerate(idx_mapas):
        t_val = t[idx_t]
        Tf = np.full_like(Xf_all, t_val)

        inp = tf.concat([
            to_tensor(escalar_x(Xf_all)),
            to_tensor(escalar_y(Yf_all)),
            to_tensor(escalar_t(Tf)),
        ], axis=1)
        Ez_pred = descalar_u(
            model(inp, training=False).numpy()
        ).reshape(Mx, My)
        Ez_real = Ez[:, :, idx_t]

        vmax = np.abs(Ez_real).max()
        ext = [x.min(), x.max(), y.min(), y.max()]
        kw = dict(origin='lower', aspect='auto', extent=ext,
                  cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        axes[0, col].imshow(Ez_real.T, **kw)
        axes[0, col].set_title(f'FDTD  t={t_val:.1e}s', fontsize=9)
        axes[1, col].imshow(Ez_pred.T, **kw)
        err = (np.linalg.norm(Ez_pred - Ez_real) /
               (np.linalg.norm(Ez_real) + 1e-12))
        axes[1, col].set_title(f'PINN  Err={err:.2f}', fontsize=9)

        for row in range(2):
            axes[row, col].set_xlabel('x (m)', fontsize=8)
            axes[row, col].set_ylabel('y (m)', fontsize=8)
            axes[row, col].plot(x[nodo_fuente], y[nodo_fuente], 'g*', ms=8)

    plt.suptitle(f'Ez(x,y) -- FDTD vs PINN | {n_sparse} pts dispersos', fontsize=13)
    plt.tight_layout()
    plt.savefig('/content/mapa_Ez_xy.png', dpi=120, bbox_inches='tight')
    plt.show()


def plot_historial_perdidas(adam_results, lbfgs_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    loss_history = adam_results['loss_history']
    data_history = adam_results['data_history']
    phys_history = adam_results['phys_history']
    epochs_adam = adam_results['epochs_adam']
    N_PRETRAIN = adam_results['N_PRETRAIN']
    lbfgs_losses = lbfgs_results['lbfgs_losses']

    axes[0].plot(loss_history, 'b-', lw=0.7, label='Adam total')
    if lbfgs_losses:
        x_lb = np.arange(epochs_adam, epochs_adam + len(lbfgs_losses))
        axes[0].plot(x_lb, lbfgs_losses, 'r-', lw=0.7, label='L-BFGS')
    axes[0].axvline(x=epochs_adam, color='g', ls='--', alpha=0.7, label='Inicio L-BFGS')
    axes[0].set(yscale='log', title='Loss Total: Adam -> L-BFGS',
                xlabel='Epochs / Iteraciones', ylabel='Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(data_history, 'b-', lw=0.7, label='Datos')
    axes[1].plot(phys_history, 'r-', lw=0.7, label='Fisica')
    axes[1].axvline(x=N_PRETRAIN, color='orange', ls='--', alpha=0.7, label='Inicio fisica')
    axes[1].set(yscale='log', title='Loss Datos vs Fisica',
                xlabel='Epochs', ylabel='Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/content/loss_history.png', dpi=120, bbox_inches='tight')
    plt.show()
