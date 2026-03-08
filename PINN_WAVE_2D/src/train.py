import time
import numpy as np
import tensorflow as tf
import scipy.optimize as sopt

from src.losses import data_loss_function, physics_loss_function, ricker_wavelet_tf


# =============================================================================
# UTILIDADES PARA L-BFGS
# =============================================================================
def get_weights(model):
    """Extrae todos los pesos del modelo como un vector numpy 1D."""
    return np.concatenate([w.numpy().flatten()
                           for w in model.trainable_variables])


def set_weights(model, flat_weights):
    """Asigna un vector 1D de pesos de vuelta al modelo."""
    idx = 0
    for var in model.trainable_variables:
        size = np.prod(var.shape)
        var.assign(tf.reshape(
            tf.cast(flat_weights[idx:idx + size], tf.float32), var.shape))
        idx += size


# -----------------------------------------------------------------------------
# Funcion monolitica para L-BFGS con termino fuente.
# Se computan L_data y L_PDE de forma inline dentro de un unico GradientTape
# externo (tape_outer). Esto garantiza que tape_outer tenga visibilidad
# completa de la cadena computacional, evitando la opacidad que introduce
# @tf.function en cintas anidadas.
# La PDE incluye el mismo termino fuente Ricker que se usa en Adam,
# garantizando consistencia entre ambas fases de optimizacion.
# -----------------------------------------------------------------------------
def loss_and_grads_lbfgs(flat_weights, model,
                         X0, Y0, T0, U0_norm, W0_time,
                         X, Y, T, k_x, k_y, k_t, c0,
                         x_src_norm, y_src_norm, sigma_src,
                         fp_tf, t0_src_tf, lb_t_phys_tf,
                         range_t_phys_tf, u_std_tf,
                         w_data=100.0, w_phys=10.0):
    set_weights(model, flat_weights)

    with tf.GradientTape() as tape_outer:
        MSEU = data_loss_function(model, X0, Y0, T0, U0_norm, W0_time)

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([X, Y, T])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([X, Y, T])
                u_pred_phys = model(tf.concat([X, Y, T], axis=1))
            u_x = tape1.gradient(u_pred_phys, X)
            u_y = tape1.gradient(u_pred_phys, Y)
            u_t = tape1.gradient(u_pred_phys, T)
        u_xx = tape2.gradient(u_x, X)
        u_yy = tape2.gradient(u_y, Y)
        u_tt = tape2.gradient(u_t, T)
        del tape1, tape2

        # -- Cálculo del término fuente (completo) --
        t_phys = T * range_t_phys_tf / 2.0 + (lb_t_phys_tf + range_t_phys_tf / 2.0)
        spatial_gauss = tf.exp(-((X - x_src_norm)**2 + (Y - y_src_norm)**2)
                               / (2.0 * sigma_src**2))
        source = ricker_wavelet_tf(t_phys, fp_tf, t0_src_tf) * spatial_gauss

        # -- Factor de escala temporal (corregido) --
        scale_t = (range_t_phys_tf / 2.0) ** 2
        source_norm = source / u_std_tf * scale_t

        lambda_x = (c0 * k_x / k_t) ** 2
        lambda_y = (c0 * k_y / k_t) ** 2
        F = u_tt - lambda_x * u_xx - lambda_y * u_yy - source_norm
        MSEF = tf.reduce_mean(tf.square(F))

        total_loss = w_data * MSEU + w_phys * MSEF

    grads = tape_outer.gradient(total_loss, model.trainable_variables)
    grads_flat = np.concatenate([g.numpy().flatten() for g in grads])
    return (total_loss.numpy().astype(np.float64),
            grads_flat.astype(np.float64))


# =============================================================================
# FASE 1 Y 2: ADAM CON BALANCEO ADAPTATIVO
# =============================================================================
def entrenar_adam(model, training_data, colocacion_data, phys_data, callback_snapshot=None):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=6000,
        decay_rate=0.90
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    epochs_adam = 50000
    loss_history = []
    data_history = []
    phys_history = []

    # Hiperparametros de entrenamiento
    W_DATA = 50
    w_phys = tf.Variable(1.0, dtype=tf.float32, trainable=False)
    CLIP_NORM = 5.0
    N_PRETRAIN = 20000
    N_RAMP = 10000
    FREQ_BALANCE = 1000
    BETA_EMA = 0.1

    X0 = training_data['X0']
    Y0 = training_data['Y0']
    T0 = training_data['T0']
    U0_norm = training_data['U0_norm']
    W0_time = training_data['W0_time']

    X = colocacion_data['X']
    Y = colocacion_data['Y']
    T = colocacion_data['T']

    k_x = phys_data['k_x']
    k_y = phys_data['k_y']
    k_t = phys_data['k_t']
    c0 = phys_data['c0']
    x_src_norm_tf = phys_data['x_src_norm_tf']
    y_src_norm_tf = phys_data['y_src_norm_tf']
    sigma_src = phys_data['sigma_src']
    fp_tf = phys_data['fp_tf']
    t0_src_tf = phys_data['t0_src_tf']
    lb_t_phys_tf = phys_data['lb_t_phys_tf']
    range_t_phys_tf = phys_data['range_t_phys_tf']
    u_std_tf = phys_data['u_std_tf']

    # Argumentos comunes para physics_loss_function (evita repeticion)
    phys_args = (k_x, k_y, k_t, c0, x_src_norm_tf, y_src_norm_tf, sigma_src,
                 fp_tf, t0_src_tf, lb_t_phys_tf, range_t_phys_tf, u_std_tf)

    print("\n" + "=" * 60)
    print("FASE 1 & 2: Entrenamiento con Adam")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(epochs_adam):

        with tf.GradientTape() as tape:
            MSEU = data_loss_function(model, X0, Y0, T0, U0_norm, W0_time)

            if epoch < N_PRETRAIN:
                total_loss = W_DATA * MSEU
                MSEF = tf.constant(0.0)
                alpha = 0.0
            else:
                alpha = min((epoch - N_PRETRAIN) / float(N_RAMP), 1.0)
                MSEF = physics_loss_function(model, X, Y, T, *phys_args)
                total_loss = W_DATA * MSEU + alpha * w_phys * MSEF

        gradients = tape.gradient(total_loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=CLIP_NORM)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_history.append(float(total_loss))
        data_history.append(float(MSEU))
        phys_history.append(float(MSEF))

        # --- Balanceo adaptativo de pesos (Wang et al. [4]) ---
        if (epoch >= N_PRETRAIN + N_RAMP
            and epoch % FREQ_BALANCE == 0
            and float(MSEF) > 0):

            with tf.GradientTape(persistent=True) as tape_bal:
                loss_d = data_loss_function(model, X0, Y0, T0, U0_norm, W0_time)
                loss_p = physics_loss_function(model, X, Y, T, *phys_args)
            grads_d = tape_bal.gradient(loss_d, model.trainable_variables)
            grads_p = tape_bal.gradient(loss_p, model.trainable_variables)
            del tape_bal

            max_grad_d = max(tf.reduce_max(tf.abs(g)).numpy()
                             for g in grads_d if g is not None)
            mean_grad_p = np.mean([tf.reduce_mean(tf.abs(g)).numpy()
                                   for g in grads_p if g is not None])

            if mean_grad_p > 1e-12:
                lambda_hat = max_grad_d / mean_grad_p
                new_w_phys = (1.0 - BETA_EMA) * w_phys.numpy() + BETA_EMA * lambda_hat
                w_phys.assign(new_w_phys)

        if epoch % 500 == 0:
            contrib_data = W_DATA * float(MSEU)
            contrib_phys = alpha * w_phys.numpy() * float(MSEF)
            ratio = contrib_phys / (contrib_data + 1e-12)
            elapsed = time.time() - start_time
            print(f"[Adam] Epoch {epoch:5d} | Loss: {float(total_loss):.4e} | "
                  f"Data: {float(MSEU):.4e} | PDE: {float(MSEF):.4e} | "
                  f"alpha: {alpha:.3f} | w_phys: {w_phys.numpy():.2f} | "
                  f"ratio: {ratio:.2f} | t: {elapsed:.0f}s")

        if callback_snapshot is not None and (epoch % 5000 == 0 or epoch == epochs_adam - 1):
            callback_snapshot(epoch, model)

    adam_time = time.time() - start_time
    print(f"\nAdam finalizado en {adam_time:.1f}s")

    return {
        'optimizer': optimizer,
        'epochs_adam': epochs_adam,
        'loss_history': loss_history,
        'data_history': data_history,
        'phys_history': phys_history,
        'W_DATA': W_DATA,
        'w_phys': w_phys,
        'N_PRETRAIN': N_PRETRAIN,
        'adam_time': adam_time,
    }


# =============================================================================
# FASE 3: L-BFGS
# =============================================================================
def entrenar_lbfgs(model, training_data, colocacion_data, phys_data, adam_results):
    # Refinamiento cuasi-Newton sobre la solucion de Adam.
    # Se usa un piso minimo de w_phys y reinicios consecutivos con tolerancias
    # mas estrictas para evitar paros prematuros y exprimir mas refinamiento.
    print("\n" + "=" * 60)
    print("FASE 3: Refinamiento con L-BFGS")
    print("=" * 60)

    final_w_data = adam_results['W_DATA']
    final_w_phys = max(float(adam_results['w_phys'].numpy()), 25.0)
    print(f"Pesos finales -> w_data={final_w_data:.2f}, w_phys={final_w_phys:.2f}")

    LBFGS_RESTARTS = 4
    LBFGS_MAXITER_PER_RESTART = 1500
    LBFGS_MAXFUN_PER_RESTART = 20000
    LBFGS_LOG_EVERY = 10
    LBFGS_MIN_REL_IMPROV = 1e-4
    LBFGS_FTOL = 1e-12
    LBFGS_GTOL = 1e-9

    X0 = training_data['X0']
    Y0 = training_data['Y0']
    T0 = training_data['T0']
    U0_norm = training_data['U0_norm']
    W0_time = training_data['W0_time']

    X = colocacion_data['X']
    Y = colocacion_data['Y']
    T = colocacion_data['T']

    k_x = phys_data['k_x']
    k_y = phys_data['k_y']
    k_t = phys_data['k_t']
    c0 = phys_data['c0']
    x_src_norm_tf = phys_data['x_src_norm_tf']
    y_src_norm_tf = phys_data['y_src_norm_tf']
    sigma_src = phys_data['sigma_src']
    fp_tf = phys_data['fp_tf']
    t0_src_tf = phys_data['t0_src_tf']
    lb_t_phys_tf = phys_data['lb_t_phys_tf']
    range_t_phys_tf = phys_data['range_t_phys_tf']
    u_std_tf = phys_data['u_std_tf']

    lbfgs_losses = []
    lbfgs_start = time.time()
    w0 = get_weights(model).astype(np.float64)
    print(f"Pesos iniciales: {w0.shape[0]} parametros")
    print("Iniciando L-BFGS...\n")

    loss_prev, _ = loss_and_grads_lbfgs(
        w0, model, X0, Y0, T0, U0_norm, W0_time, X, Y, T, k_x, k_y, k_t, c0,
        x_src_norm_tf, y_src_norm_tf, sigma_src,
        fp_tf, t0_src_tf, lb_t_phys_tf, range_t_phys_tf, u_std_tf,
        w_data=final_w_data, w_phys=final_w_phys
    )
    print(f"Loss inicial L-BFGS: {float(loss_prev):.4e}")

    w_current = w0
    total_nit = 0
    total_nfev = 0
    result = None

    for restart in range(1, LBFGS_RESTARTS + 1):
        iter_local = [0]
        restart_start = time.time()
        print(f"\n[L-BFGS] Reinicio {restart}/{LBFGS_RESTARTS}")

        def callback_lbfgs(flat_weights):
            iter_local[0] += 1
            loss_val, _ = loss_and_grads_lbfgs(
                flat_weights, model, X0, Y0, T0, U0_norm, W0_time,
                X, Y, T, k_x, k_y, k_t, c0,
                x_src_norm_tf, y_src_norm_tf, sigma_src,
                fp_tf, t0_src_tf, lb_t_phys_tf, range_t_phys_tf, u_std_tf,
                w_data=final_w_data, w_phys=final_w_phys
            )
            lbfgs_losses.append(float(loss_val))
            if iter_local[0] == 1 or iter_local[0] % LBFGS_LOG_EVERY == 0:
                elapsed = time.time() - lbfgs_start
                print(f"[L-BFGS r{restart}] Iter {iter_local[0]:4d} | "
                      f"Loss: {loss_val:.4e} | t: {elapsed:.1f}s")

        result = sopt.minimize(
            fun=loss_and_grads_lbfgs,
            x0=w_current,
            args=(model, X0, Y0, T0, U0_norm, W0_time, X, Y, T, k_x, k_y, k_t, c0,
                  x_src_norm_tf, y_src_norm_tf, sigma_src,
                  fp_tf, t0_src_tf, lb_t_phys_tf, range_t_phys_tf, u_std_tf,
                  final_w_data, final_w_phys),
            method='L-BFGS-B',
            jac=True,
            callback=callback_lbfgs,
            options={
                'maxiter': LBFGS_MAXITER_PER_RESTART,
                'maxfun': LBFGS_MAXFUN_PER_RESTART,
                'ftol': LBFGS_FTOL,
                'gtol': LBFGS_GTOL,
                'maxls': 100,
                'maxcor': 50,
                'disp': False,
            }
        )

        set_weights(model, result.x)
        w_current = result.x
        total_nit += int(result.nit)
        total_nfev += int(result.nfev)

        loss_new = float(result.fun)
        rel_improv = (float(loss_prev) - loss_new) / (abs(float(loss_prev)) + 1e-12)
        restart_elapsed = time.time() - restart_start
        print(f"   nit={result.nit} | nfev={result.nfev} | "
              f"loss={loss_new:.4e} | mejora_rel={rel_improv:.3e} | "
              f"exito={result.success} | t={restart_elapsed:.1f}s")
        print(f"   Mensaje: {result.message}")

        if rel_improv < LBFGS_MIN_REL_IMPROV:
            print("   Mejora marginal detectada; se detiene L-BFGS.")
            break
        loss_prev = loss_new

    lbfgs_time = time.time() - lbfgs_start
    print(f"\nL-BFGS finalizado en {lbfgs_time:.1f}s")
    print(f"   Iteraciones (total)       : {total_nit}")
    print(f"   Evaluaciones func (total) : {total_nfev}")
    if result is not None:
        print(f"   Loss final                : {result.fun:.4e}")
        print(f"   Convergio (ultimo run)    : {result.success}")
        print(f"   Mensaje (ultimo run)      : {result.message}")

    return {
        'result': result,
        'lbfgs_losses': lbfgs_losses,
        'lbfgs_time': lbfgs_time,
        'total_nit': total_nit,
        'total_nfev': total_nfev,
    }
