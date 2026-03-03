import numpy as np
import tensorflow as tf
from pyDOE3 import lhs


def dataloader(path):
    # Load data from file txt format 
    data = np.loadtxt(path)

    espacio = data[1:, 0]
    tiempo = data[0, 1:]
    EzenXyT = data[1:, 1:]
    return espacio, tiempo, EzenXyT


def escalar_x(v, lb_x, ub_x):
    # Scale the spatial coordinate to the interval [-1,1]
    return 2.0 * (v - lb_x) / (ub_x - lb_x) - 1.0

   
def escalar_t(v, lb_t, ub_t):
    # Scale the temporal coordinate to the interval [-1,1]
    return 2.0 * (v - lb_t) / (ub_t - lb_t) - 1.0


def escalar_u(v, u_mean, u_std):
    # Normalize the physical field using mean and standard deviation
    return (v - u_mean) / u_std


def descalar_u(v, u_mean, u_std):
    # Recover the physical field 
    return v * u_std + u_mean


def to_tensor(arr):
    # Convert to tensor 
    return tf.convert_to_tensor(arr, dtype=tf.float32)

# Prepare training data from FDTD field
def preparar_datos_entrenamiento(path, n_sparse=150, source_x=1.5):
    # Build training tensors from FDTD Field 
    espacio, tiempo, EzenXyT = dataloader(path)

    # Define the grid size
    delta_x = (espacio[1] - espacio[0]).item()
    delta_t = (tiempo[1] - tiempo[0]).item()
    # Define the number of rows and columns 
    renglon, columna = EzenXyT.shape
    
    # Define the number of boundary condition points
    numDatosBCExtr = int(np.round(0.8 * columna))
    idx_pool = np.arange(1, columna)
    idx_ext = np.random.choice(idx_pool, numDatosBCExtr, replace=False)

    # Define the boundary condition points
    t0BC_left = tiempo[idx_ext].reshape(-1, 1)
    x0BC_left = np.full((numDatosBCExtr, 1), espacio[0])
    u0BC_left = np.zeros((numDatosBCExtr, 1))

    t0BC_right = tiempo[idx_ext].reshape(-1, 1)
    x0BC_right = np.full((numDatosBCExtr, 1), espacio[-1])
    u0BC_right = np.zeros((numDatosBCExtr, 1))

    # Define the number of initial condition points 
    numDatosIC = int(np.round(0.8 * renglon))
    idx_x_ic = np.random.choice(renglon, numDatosIC, replace=False)
    # Define the initial condition points
    x0IC = espacio[idx_x_ic].reshape(-1, 1)
    t0IC = np.zeros((numDatosIC, 1))
    u0IC = EzenXyT[idx_x_ic, 0].reshape(-1, 1)

    # Define the node of the source
    nodo_fuente = int(np.round(source_x / delta_x)) - 1
    x_fuente_arr = np.full((columna, 1), espacio[nodo_fuente])
    t_fuente_arr = tiempo.reshape(-1, 1)
    u_fuente_arr = EzenXyT[nodo_fuente, :].reshape(-1, 1)

    # Define the number of sparse points
    idx_x_rand = np.random.choice(renglon, n_sparse, replace=True)
    idx_t_rand = np.random.choice(columna, n_sparse, replace=True)
    x_sparse = espacio[idx_x_rand].reshape(-1, 1)
    t_sparse = tiempo[idx_t_rand].reshape(-1, 1)
    u_sparse = EzenXyT[idx_x_rand, idx_t_rand].reshape(-1, 1)

    # Define the min and max of the spatial and temporal coordinates
    lb_x = espacio.min() ; ub_x = espacio.max()
    lb_t = tiempo.min() ; ub_t = tiempo.max()

    # obtain the mean and standard deviation of the physical field
    u_mean = EzenXyT.mean()
    u_std = EzenXyT.std()

    # Concatenate the training data 
    X0_phys = np.concatenate(
        [x_sparse, x0IC, x0BC_left, x0BC_right, x_fuente_arr],
        axis=0,
    )
    T0_phys = np.concatenate(
        [t_sparse, t0IC, t0BC_left, t0BC_right, t_fuente_arr],
        axis=0,
    )
    U0 = np.concatenate(
        [u_sparse, u0IC, u0BC_left, u0BC_right, u_fuente_arr],
        axis=0,
    )

    # Convert to tensor 
    X0 = to_tensor(escalar_x(X0_phys, lb_x, ub_x))
    T0 = to_tensor(escalar_t(T0_phys, lb_t, ub_t))
    U0_norm = to_tensor(escalar_u(U0, u_mean, u_std))

    return {
        "espacio": espacio,
        "tiempo": tiempo,
        "EzenXyT": EzenXyT,
        "delta_x": delta_x,
        "delta_t": delta_t,
        "renglon": renglon,
        "columna": columna,
        "t0BC_left": t0BC_left,
        "x0BC_left": x0BC_left,
        "u0BC_left": u0BC_left,
        "t0BC_right": t0BC_right,
        "x0BC_right": x0BC_right,
        "u0BC_right": u0BC_right,
        "x0IC": x0IC,
        "t0IC": t0IC,
        "u0IC": u0IC,
        "nodo_fuente": nodo_fuente,
        "x_fuente_arr": x_fuente_arr,
        "t_fuente_arr": t_fuente_arr,
        "u_fuente_arr": u_fuente_arr,
        "x_sparse": x_sparse,
        "t_sparse": t_sparse,
        "u_sparse": u_sparse,
        "lb_x": lb_x,
        "ub_x": ub_x,
        "lb_t": lb_t,
        "ub_t": ub_t,
        "u_mean": u_mean,
        "u_std": u_std,
        "X0_phys": X0_phys,
        "T0_phys": T0_phys,
        "U0": U0,
        "X0": X0,
        "T0": T0,
        "U0_norm": U0_norm,
    }


def preparar_puntos_colocacion(
    espacio,
    delta_x,
    lb_x,
    ub_x,
    n_colocation=50000,
    source_x=1.5,
):
    # Generate points of colocation using Latin Hypercube Sampling
    lb_col = np.array([-1.0, -1.0])
    ub_col = np.array([1.0, 1.0])

    colocation = lhs(2, samples=n_colocation)
    colocation_s = lb_col + (ub_col - lb_col) * colocation

    x_fuente_norm = float(escalar_x(np.array([source_x]), lb_x, ub_x))
    delta_exclusion = 3.0 * delta_x * (2.0 / (ub_x - lb_x))
    mascara = np.abs(colocation_s[:, 0] - x_fuente_norm) > delta_exclusion
    colocation_s = colocation_s[mascara]

    X = to_tensor(colocation_s[:, 0:1])
    T = to_tensor(colocation_s[:, 1:2])

    return {
        "colocation_s": colocation_s,
        "X": X,
        "T": T,
    }
