import numpy as np
import scipy.io as sio
import tensorflow as tf
from pyDOE3 import lhs


# =============================================================================
# CARGA DE DATOS Y PREPARACION
# =============================================================================
def cargar_datos_fdtd(path='/content/Resultados_Simulacion.mat'):
    data = sio.loadmat(path)

    x = data['x'].flatten()
    y = data['y'].flatten()
    t = data['t'].flatten()
    Ez = data['EzHxHyenXT']

    Mx, My, Q = Ez.shape
    delta_x = (x[1] - x[0]).item()
    delta_y = (y[1] - y[0]).item()
    delta_t = (t[1] - t[0]).item()

    print(f"x_arr : {x.shape}  ->  [{x[0]:.4f}, {x[-1]:.4f}] m")
    print(f"y_arr : {y.shape}  ->  [{y[0]:.4f}, {y[-1]:.4f}] m")
    print(f"t_arr : {t.shape}  ->  [{t[0]:.4e}, {t[-1]:.4e}] s")
    print(f"Ez    : {Ez.shape}  ->  min={Ez.min():.3e}, max={Ez.max():.3e}")

    # Parametros de la fuente (del FDTD de MATLAB)
    fp = 1e9             # frecuencia central del Ricker
    tau = 1.0 / fp       # = 1e-9 s
    t0_src = tau         # desfasamiento temporal de la fuente

    return {
        'x': x,
        'y': y,
        't': t,
        'Ez': Ez,
        'Mx': Mx,
        'My': My,
        'Q': Q,
        'delta_x': delta_x,
        'delta_y': delta_y,
        'delta_t': delta_t,
        'fp': fp,
        'tau': tau,
        't0_src': t0_src,
    }


# =============================================================================
# DATOS DE ENTRENAMIENTO
# =============================================================================
def preparar_datos_entrenamiento(fdtd_data):
    x = fdtd_data['x']
    y = fdtd_data['y']
    t = fdtd_data['t']
    Ez = fdtd_data['Ez']
    Mx = fdtd_data['Mx']
    My = fdtd_data['My']
    Q = fdtd_data['Q']

    # --- Condiciones de frontera (PEC: Ez=0 en los 4 bordes) ---
    n_bc = int(np.round(0.8 * Mx))
    idx_t_bc = np.random.choice(Q, n_bc, replace=True)
    idx_x_bc = np.random.choice(Mx, n_bc, replace=False)
    idx_y_bc = np.random.choice(My, n_bc, replace=False)

    x_bc_x0 = np.zeros((n_bc, 1))
    y_bc_x0 = y[idx_y_bc].reshape(-1, 1)
    t_bc_x0 = t[idx_t_bc].reshape(-1, 1)
    u_bc_x0 = np.zeros((n_bc, 1))

    x_bc_xmax = np.full((n_bc, 1), x[-1])
    y_bc_xmax = y[idx_y_bc].reshape(-1, 1)
    t_bc_xmax = t[idx_t_bc].reshape(-1, 1)
    u_bc_xmax = np.zeros((n_bc, 1))

    x_bc_y0 = x[idx_x_bc].reshape(-1, 1)
    y_bc_y0 = np.zeros((n_bc, 1))
    t_bc_y0 = t[idx_t_bc].reshape(-1, 1)
    u_bc_y0 = np.zeros((n_bc, 1))

    x_bc_ymax = x[idx_x_bc].reshape(-1, 1)
    y_bc_ymax = np.full((n_bc, 1), y[-1])
    t_bc_ymax = t[idx_t_bc].reshape(-1, 1)
    u_bc_ymax = np.zeros((n_bc, 1))

    # --- Condicion inicial: t = t[0] (primer paso temporal FDTD) ---
    numDatosIC = int(np.round(0.8 * Mx))
    idx_x_ic = np.random.choice(Mx, numDatosIC, replace=False)
    idx_y_ic = np.random.choice(My, numDatosIC, replace=False)

    x0IC = x[idx_x_ic].reshape(-1, 1)
    y0IC = y[idx_y_ic].reshape(-1, 1)
    t0IC = np.full((numDatosIC, 1), t[0])
    u0IC = Ez[idx_x_ic, idx_y_ic, 0].reshape(-1, 1)

    # --- Datos en la fuente ---
    # Correccion de indexado: round(M/2) en MATLAB (1-based) = Mx//2 - 1 (0-based)
    nodo_fuente = Mx // 2 - 1
    x_fuente_arr = np.full((Q, 1), x[nodo_fuente])
    y_fuente_arr = np.full((Q, 1), y[nodo_fuente])
    t_fuente_arr = t.reshape(-1, 1)
    u_fuente_arr = Ez[nodo_fuente, nodo_fuente, :].reshape(-1, 1)

    print(f"Nodo de la fuente: {nodo_fuente} | x = {x[nodo_fuente]:.4f} m")

    # --- Puntos dispersos interiores ---
    # Muestreo estratificado por cuadrante con ponderacion temporal mixta.
    # Se refuerza el inicio para mejorar el ajuste temprano sin perder
    # representatividad en tiempos medios/tardios.
    n_sparse = 20000
    n_cada = n_sparse // 4

    idx_x_izq = np.where(x <= x[nodo_fuente])[0]
    idx_x_der = np.where(x > x[nodo_fuente])[0]
    idx_y_inf = np.where(y <= y[nodo_fuente])[0]
    idx_y_sup = np.where(y > y[nodo_fuente])[0]

    ix_q1 = np.random.choice(idx_x_der, n_cada, replace=True)
    iy_q1 = np.random.choice(idx_y_sup, n_cada, replace=True)
    ix_q2 = np.random.choice(idx_x_izq, n_cada, replace=True)
    iy_q2 = np.random.choice(idx_y_sup, n_cada, replace=True)
    ix_q3 = np.random.choice(idx_x_izq, n_cada, replace=True)
    iy_q3 = np.random.choice(idx_y_inf, n_cada, replace=True)
    ix_q4 = np.random.choice(idx_x_der, n_cada, replace=True)
    iy_q4 = np.random.choice(idx_y_inf, n_cada, replace=True)

    t_frac_sparse = np.linspace(0.0, 1.0, Q)
    w_early_sparse = 1.0 + 2.5 * (1.0 - t_frac_sparse) ** 2
    w_late_sparse = 1.0 + 0.8 * t_frac_sparse
    t_weights = 0.6 * w_early_sparse + 0.4 * w_late_sparse
    t_weights /= t_weights.sum()
    it_q1 = np.random.choice(Q, n_cada, replace=True, p=t_weights)
    it_q2 = np.random.choice(Q, n_cada, replace=True, p=t_weights)
    it_q3 = np.random.choice(Q, n_cada, replace=True, p=t_weights)
    it_q4 = np.random.choice(Q, n_cada, replace=True, p=t_weights)

    x_sparse = np.concatenate([x[ix_q1], x[ix_q2], x[ix_q3], x[ix_q4]]).reshape(-1, 1)
    y_sparse = np.concatenate([y[iy_q1], y[iy_q2], y[iy_q3], y[iy_q4]]).reshape(-1, 1)
    t_sparse = np.concatenate([t[it_q1], t[it_q2], t[it_q3], t[it_q4]]).reshape(-1, 1)
    u_sparse = np.concatenate([
        Ez[ix_q1, iy_q1, it_q1],
        Ez[ix_q2, iy_q2, it_q2],
        Ez[ix_q3, iy_q3, it_q3],
        Ez[ix_q4, iy_q4, it_q4]
    ]).reshape(-1, 1)

    print(f"Dispersos por cuadrante : {n_cada} puntos")
    print(f"Dispersos total         : {x_sparse.shape[0]} puntos")
    print(f"u range                 : [{u_sparse.min():.3e}, {u_sparse.max():.3e}]")

    # =============================================================================
    # ESCALAMIENTO
    # Min-max -> [-1, 1] para coordenadas espaciotemporales.
    # Estandarizacion (mean/std) para el campo Ez.
    # =============================================================================
    lb_x = x.min()
    ub_x = x.max()
    lb_y = y.min()
    ub_y = y.max()
    lb_t = t.min()
    ub_t = t.max()

    def escalar_x(v):
        return 2.0 * (v - lb_x) / (ub_x - lb_x) - 1.0

    def escalar_y(v):
        return 2.0 * (v - lb_y) / (ub_y - lb_y) - 1.0

    def escalar_t(v):
        return 2.0 * (v - lb_t) / (ub_t - lb_t) - 1.0

    u_mean = Ez.mean()
    u_std = Ez.std()

    def escalar_u(v):
        return (v - u_mean) / u_std

    def descalar_u(v):
        return v * u_std + u_mean

    def to_tensor(arr):
        return tf.convert_to_tensor(arr, dtype=tf.float32)

    X0_phys = np.concatenate([x_sparse, x0IC, x_bc_x0, x_bc_xmax,
                              x_bc_y0, x_bc_ymax, x_fuente_arr], axis=0)
    Y0_phys = np.concatenate([y_sparse, y0IC, y_bc_x0, y_bc_xmax,
                              y_bc_y0, y_bc_ymax, y_fuente_arr], axis=0)
    T0_phys = np.concatenate([t_sparse, t0IC, t_bc_x0, t_bc_xmax,
                              t_bc_y0, t_bc_ymax, t_fuente_arr], axis=0)
    U0 = np.concatenate([u_sparse, u0IC, u_bc_x0, u_bc_xmax,
                         u_bc_y0, u_bc_ymax, u_fuente_arr], axis=0)

    # Mayor peso a tiempos tempranos en L_data para atacar el error inicial.
    EARLY_DATA_GAIN = 2.5
    EARLY_DATA_POWER = 2.0
    t_frac_data = (T0_phys - lb_t) / (ub_t - lb_t + 1e-12)
    w_data_time_np = 1.0 + EARLY_DATA_GAIN * (1.0 - t_frac_data) ** EARLY_DATA_POWER
    w_data_time_np = w_data_time_np.astype(np.float32)

    X0 = to_tensor(escalar_x(X0_phys))
    Y0 = to_tensor(escalar_y(Y0_phys))
    T0 = to_tensor(escalar_t(T0_phys))
    U0_norm = to_tensor(escalar_u(U0))
    W0_time = to_tensor(w_data_time_np)

    total_puntos = X0.shape[0]
    total_dominio = Mx * My * Q
    print(f"X0      : {X0.shape}  [{X0.numpy().min():.2f}, {X0.numpy().max():.2f}]")
    print(f"Y0      : {Y0.shape}  [{Y0.numpy().min():.2f}, {Y0.numpy().max():.2f}]")
    print(f"T0      : {T0.shape}  [{T0.numpy().min():.2f}, {T0.numpy().max():.2f}]")
    print(f"U0_norm : {U0_norm.shape}  [{U0_norm.numpy().min():.2f}, {U0_norm.numpy().max():.2f}]")
    print(f"W0_time : {W0_time.shape}  [{W0_time.numpy().min():.2f}, {W0_time.numpy().max():.2f}]")
    print(f"\nTotal puntos : {total_puntos}")
    print(f"% del dominio: {100 * total_puntos / total_dominio:.4f}%")

    return {
        **fdtd_data,
        'nodo_fuente': nodo_fuente,
        'n_sparse': n_sparse,
        'n_cada': n_cada,
        'x_sparse': x_sparse,
        'y_sparse': y_sparse,
        't_sparse': t_sparse,
        'u_sparse': u_sparse,
        'lb_x': lb_x,
        'ub_x': ub_x,
        'lb_y': lb_y,
        'ub_y': ub_y,
        'lb_t': lb_t,
        'ub_t': ub_t,
        'u_mean': u_mean,
        'u_std': u_std,
        'escalar_x': escalar_x,
        'escalar_y': escalar_y,
        'escalar_t': escalar_t,
        'escalar_u': escalar_u,
        'descalar_u': descalar_u,
        'to_tensor': to_tensor,
        'X0': X0,
        'Y0': Y0,
        'T0': T0,
        'U0_norm': U0_norm,
        'W0_time': W0_time,
        'X0_phys': X0_phys,
        'Y0_phys': Y0_phys,
        'T0_phys': T0_phys,
        'U0': U0,
    }


# =============================================================================
# PUNTOS DE COLOCACION
# =============================================================================
# Distribucion bipartita: interior [-1,1]^3 + exterior [-3,3]^3.
# La extension del dominio de colocacion permite que la PDE actue como
# regularizador fuera de la region supervisada, forzando comportamiento
# fisicamente consistente en la zona de extrapolacion [2].
# =============================================================================
def preparar_puntos_colocacion(training_data):
    n_interior = 35000
    n_exterior = 35000

    col_int = -1.0 + 2.0 * lhs(3, samples=n_interior)

    lb_ext = np.array([-3.0, -3.0, -3.0])
    ub_ext = np.array([3.0, 3.0, 3.0])
    col_ext = lb_ext + (ub_ext - lb_ext) * lhs(3, samples=n_exterior)

    colocation_s = np.concatenate([col_int, col_ext], axis=0)

    # Zona de exclusion alrededor de la fuente: ~1 longitud de onda (20 celdas).
    # Aunque el termino fuente esta modelado explicitamente, se mantiene una
    # zona de exclusion moderada para estabilidad numerica en la vecindad
    # inmediata de la singularidad.
    x = training_data['x']
    y = training_data['y']
    nodo_fuente = training_data['nodo_fuente']
    escalar_x = training_data['escalar_x']
    escalar_y = training_data['escalar_y']
    delta_x = training_data['delta_x']
    delta_y = training_data['delta_y']
    ub_x = training_data['ub_x']
    lb_x = training_data['lb_x']
    ub_y = training_data['ub_y']
    lb_y = training_data['lb_y']

    x_fuente_norm = float(escalar_x(np.array(x[nodo_fuente])))
    y_fuente_norm = float(escalar_y(np.array(y[nodo_fuente])))
    delta_exclusion_x = 3.0 * delta_x * (2.0 / (ub_x - lb_x))
    delta_exclusion_y = 3.0 * delta_y * (2.0 / (ub_y - lb_y))
    mascara = ((np.abs(colocation_s[:, 0] - x_fuente_norm) > delta_exclusion_x) |
               (np.abs(colocation_s[:, 1] - y_fuente_norm) > delta_exclusion_y))
    colocation_s = colocation_s[mascara]

    print(f"\nPuntos de colocacion fisica: {colocation_s.shape[0]}")

    to_tensor = training_data['to_tensor']
    X = to_tensor(colocation_s[:, 0:1])
    Y = to_tensor(colocation_s[:, 1:2])
    T = to_tensor(colocation_s[:, 2:3])

    return {
        'n_interior': n_interior,
        'n_exterior': n_exterior,
        'colocation_s': colocation_s,
        'X': X,
        'Y': Y,
        'T': T,
        'x_fuente_norm': x_fuente_norm,
        'y_fuente_norm': y_fuente_norm,
    }


# =============================================================================
# CONSTANTES FISICAS
# =============================================================================
def preparar_constantes_fisicas(training_data, colocacion_data):
    e0 = 8.8541e-12
    m0 = 4 * np.pi * 1e-7
    c0 = tf.cast(1.0 / np.sqrt(e0 * m0), dtype=tf.float32)

    ub_x = training_data['ub_x']
    lb_x = training_data['lb_x']
    ub_y = training_data['ub_y']
    lb_y = training_data['lb_y']
    ub_t = training_data['ub_t']
    lb_t = training_data['lb_t']
    delta_x = training_data['delta_x']
    u_std = training_data['u_std']
    fp = training_data['fp']
    t0_src = training_data['t0_src']

    k_x = tf.constant(2.0 / (ub_x - lb_x), dtype=tf.float32)
    k_y = tf.constant(2.0 / (ub_y - lb_y), dtype=tf.float32)
    k_t = tf.constant(2.0 / (ub_t - lb_t), dtype=tf.float32)

    # Parametros de la fuente en coordenadas normalizadas (tensores para @tf.function)
    x_src_norm_tf = tf.constant(colocacion_data['x_fuente_norm'], dtype=tf.float32)
    y_src_norm_tf = tf.constant(colocacion_data['y_fuente_norm'], dtype=tf.float32)
    sigma_src = tf.constant(3.0 * delta_x * (2.0 / (ub_x - lb_x)), dtype=tf.float32)
    lb_t_phys_tf = tf.constant(lb_t, dtype=tf.float32)
    range_t_phys_tf = tf.constant(ub_t - lb_t, dtype=tf.float32)
    u_std_tf = tf.constant(u_std, dtype=tf.float32)
    fp_tf = tf.constant(fp, dtype=tf.float32)
    t0_src_tf = tf.constant(t0_src, dtype=tf.float32)

    lambda_sq_x = (c0.numpy() * k_x.numpy() / k_t.numpy()) ** 2
    lambda_sq_y = (c0.numpy() * k_y.numpy() / k_t.numpy()) ** 2
    print(f"\nc0          = {c0.numpy():.4e} m/s")
    print(f"k_x         = {k_x.numpy():.4e}")
    print(f"k_y         = {k_y.numpy():.4e}")
    print(f"k_t         = {k_t.numpy():.4e}")
    print(f"lambda_sq_x = {lambda_sq_x:.4f}")
    print(f"lambda_sq_y = {lambda_sq_y:.4f}")
    print(f"sigma_src   = {sigma_src.numpy():.4e} (norm)")
    print(f"fp          = {fp:.2e} Hz")
    print(f"t0_src      = {t0_src:.2e} s")

    return {
        'e0': e0,
        'm0': m0,
        'c0': c0,
        'k_x': k_x,
        'k_y': k_y,
        'k_t': k_t,
        'x_src_norm_tf': x_src_norm_tf,
        'y_src_norm_tf': y_src_norm_tf,
        'sigma_src': sigma_src,
        'lb_t_phys_tf': lb_t_phys_tf,
        'range_t_phys_tf': range_t_phys_tf,
        'u_std_tf': u_std_tf,
        'fp_tf': fp_tf,
        't0_src_tf': t0_src_tf,
        'lambda_sq_x': lambda_sq_x,
        'lambda_sq_y': lambda_sq_y,
    }
