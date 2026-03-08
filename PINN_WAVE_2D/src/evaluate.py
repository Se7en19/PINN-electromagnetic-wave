import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def visualizar_snapshot(model, x, y, t, Ez, idx_t,
                        escalar_x, escalar_y, escalar_t, descalar_u,
                        titulo='', guardar=None):
    """
    Genera comparacion visual FDTD vs PINN en un instante t fijo.
    Emplea error absoluto cuando la norma del campo real es insignificante
    respecto a la energia total de la simulacion, evitando metricas
    relativas artificialmente infladas.
    """
    Mx_l, My_l = len(x), len(y)
    t_val = t[idx_t]

    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    Xf = Xg.reshape(-1, 1)
    Yf = Yg.reshape(-1, 1)
    Tf = np.full_like(Xf, t_val)

    inp = tf.concat([
        tf.convert_to_tensor(escalar_x(Xf), dtype=tf.float32),
        tf.convert_to_tensor(escalar_y(Yf), dtype=tf.float32),
        tf.convert_to_tensor(escalar_t(Tf), dtype=tf.float32),
    ], axis=1)

    Ez_pred = descalar_u(
        model(inp, training=False).numpy()
    ).reshape(Mx_l, My_l)
    Ez_real = Ez[:, :, idx_t]

    vmax = np.abs(Ez_real).max()
    ext = [x.min(), x.max(), y.min(), y.max()]
    kw_r = dict(origin='lower', aspect='auto', extent=ext,
                cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    kw_e = dict(origin='lower', aspect='auto', extent=ext, cmap='hot_r')

    norm_real = np.linalg.norm(Ez_real)
    norm_total = np.linalg.norm(Ez)
    if norm_real < 1e-4 * norm_total:
        rel_err = np.linalg.norm(Ez_pred - Ez_real)
        err_type = 'abs'
    else:
        rel_err = np.linalg.norm(Ez_pred - Ez_real) / norm_real
        err_type = 'rel'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(Ez_real.T, **kw_r)
    im1 = axes[1].imshow(Ez_pred.T, **kw_r)
    im2 = axes[2].imshow(np.abs(Ez_real - Ez_pred).T, **kw_e)

    for ax, lbl in zip(axes, ['FDTD real', 'PINN', '|Error|']):
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(lbl)
        ax.plot(x[len(x) // 2], y[len(y) // 2], 'g*', ms=10, label='Fuente')

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])
    plt.colorbar(im2, ax=axes[2])

    suptitle = f't = {t_val:.2e} s  |  Err_{err_type} = {rel_err:.3f}'
    if titulo:
        suptitle = titulo + ' | ' + suptitle
    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=120, bbox_inches='tight')
    plt.show()
    return rel_err


def validar_modelo_final(model, training_data):
    print("\nValidacion final...")

    x = training_data['x']
    y = training_data['y']
    t = training_data['t']
    Ez = training_data['Ez']
    escalar_x = training_data['escalar_x']
    escalar_y = training_data['escalar_y']
    escalar_t = training_data['escalar_t']
    descalar_u = training_data['descalar_u']

    Q = training_data['Q']
    indices_val = [int(Q * f) for f in [0.1, 0.25, 0.5, 0.75, 0.9]]
    errores_final = []

    for idx_t in indices_val:
        err = visualizar_snapshot(
            model, x, y, t, Ez, idx_t,
            escalar_x, escalar_y, escalar_t, descalar_u,
            titulo='Adam + L-BFGS',
            guardar=f'/content/validacion_t{idx_t}.png'
        )
        errores_final.append(err)
        print(f"  t={t[idx_t]:.2e}s | Err_rel={err:.4f}")

    error_promedio = np.mean(errores_final)
    print(f"\nError relativo promedio: {error_promedio:.4f}")

    return {
        'indices_val': indices_val,
        'errores_final': errores_final,
        'error_promedio': error_promedio,
    }
