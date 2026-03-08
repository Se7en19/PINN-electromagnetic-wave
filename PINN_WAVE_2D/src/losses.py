import numpy as np
import tensorflow as tf


# =============================================================================
# TERMINO FUENTE: Pulso de Ricker
# =============================================================================
def ricker_wavelet_tf(t_physical, fp, t0_src):
    """Pulso de Ricker: r(t) = (1 - 2(pi*fp*(t-t0))^2) * exp(-(pi*fp*(t-t0))^2)"""
    arg = np.pi * fp * (t_physical - t0_src)
    return (1.0 - 2.0 * arg**2) * tf.exp(-arg**2)


# =============================================================================
# FUNCIONES DE PERDIDA
# =============================================================================
# Residuo de la PDE con termino fuente:
#   u_tt - c^2*k_x^2/k_t^2 * u_xx - c^2*k_y^2/k_t^2 * u_yy = S(x,y,t)/u_std
# donde S = Gaussiana_espacial * Ricker_temporal.
# Las derivadas se obtienen mediante diferenciacion automatica con
# GradientTape persistente [Raissi et al., 2019].
# =============================================================================
@tf.function
def physics_loss_function(model, X, Y, T, k_x, k_y, k_t, c0,
                          x_src_norm, y_src_norm, sigma_src,
                          fp, t0_src, lb_t_phys, range_t_phys, u_std_tf):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([X, Y, T])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([X, Y, T])
            u_pred = model(tf.concat([X, Y, T], axis=1))
        u_x = tape1.gradient(u_pred, X)
        u_y = tape1.gradient(u_pred, Y)
        u_t = tape1.gradient(u_pred, T)
    u_xx = tape2.gradient(u_x, X)
    u_yy = tape2.gradient(u_y, Y)
    u_tt = tape2.gradient(u_t, T)
    del tape1, tape2

    # -- Cálculo del término fuente (completo) --
    t_phys = T * range_t_phys / 2.0 + (lb_t_phys + range_t_phys / 2.0)
    spatial_gauss = tf.exp(-((X - x_src_norm)**2 + (Y - y_src_norm)**2)
                           / (2.0 * sigma_src**2))
    source = ricker_wavelet_tf(t_phys, fp, t0_src) * spatial_gauss

    # -- Factor de escala temporal (corregido) --
    scale_t = (range_t_phys / 2.0) ** 2
    source_norm = source / u_std_tf * scale_t

    lambda_x = (c0 * k_x / k_t) ** 2
    lambda_y = (c0 * k_y / k_t) ** 2
    F = u_tt - lambda_x * u_xx - lambda_y * u_yy - source_norm
    return tf.reduce_mean(tf.square(F))


@tf.function
def data_loss_function(model, x_data, y_data, t_data, u_data, sample_weights=None):
    u_pred = model(tf.concat([x_data, y_data, t_data], axis=1))
    sq_err = tf.square(u_pred - u_data)
    if sample_weights is None:
        return tf.reduce_mean(sq_err)
    w = tf.cast(sample_weights, tf.float32)
    w = w / (tf.reduce_mean(w) + 1e-12)
    return tf.reduce_mean(w * sq_err)
