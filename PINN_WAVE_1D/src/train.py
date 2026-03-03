import numpy as np
import tensorflow as tf
import time
import scipy.optimize as sopt

from src.losses import data_loss_function, physics_loss_function


def get_weights(model):
    # flatten the weights of the model
    return np.concatenate([w.numpy().flatten() for w in model.trainable_variables])


def set_weights(model, flat_weights):
    # Restore the trainable parameters from a flat vector
    idx = 0
    for var in model.trainable_variables:
        size = np.prod(var.shape)
        var.assign(
            tf.reshape(tf.cast(flat_weights[idx:idx + size], tf.float32), var.shape)
        )
        idx += size


def loss_and_grads_lbfgs(
    flat_weights,
    model,
    X0,
    T0,
    U0_norm,
    X,
    T,
    k_x,
    k_t,
    c0,
    w_data=100.0,
    w_phys=1.0,
):
    # Return the total loss and the gradients in the required format for SciPy
    set_weights(model, flat_weights)

    with tf.GradientTape() as tape:
        MSEU = data_loss_function(model, X0, T0, U0_norm)
        MSEF = physics_loss_function(model, X, T, k_x, k_t, c0)
        total_loss = w_data * MSEU + w_phys * MSEF

    grads = tape.gradient(total_loss, model.trainable_variables)
    grads_flat = np.concatenate([g.numpy().flatten() for g in grads])

    return (
        total_loss.numpy().astype(np.float64),
        grads_flat.astype(np.float64),
    )


def crear_optimizador_adam(lr_adam=1e-3):
    # Build the Adam optimizer with exponential decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr_adam,
        decay_steps=7000,
        decay_rate=0.90,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer

# Train the model using Adam optimizer
def entrenar_adam(
    model,
    X0,
    T0,
    U0_norm,
    X,
    T,
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
    callback_epoch=None,
):
    # Execute the traning in two phases using Adam 
    optimizer = crear_optimizador_adam(lr_adam=lr_adam)

    loss_history = []
    data_history = []
    phys_history = []

    print("\n" + "=" * 60)
    print("FASE 1 & 2: Entrenamiento con Adam")
    print("=" * 60)
    start_time = time.time()
    # B. Mosley et al. 2020 using curriculum learning technique
    for epoch in range(epochs_adam):
        with tf.GradientTape() as tape:
            MSEU = data_loss_function(model, X0, T0, U0_norm)
            # If the epoch is before the physical training phase, use the data loss only
            if epoch < epoch_inicio_fisica:
                total_loss = w_data * MSEU
                MSEF = tf.constant(0.0)
                alpha = 0.0
            # If the epoch is after the physical training phase, use the data loss and the physics loss
            else:
                alpha = min((epoch - epoch_inicio_fisica) / ramp_fisica, 1.0)
                MSEF = physics_loss_function(model, X, T, k_x, k_t, c0)
                total_loss = w_data * MSEU + alpha * w_phys * MSEF

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_history.append(float(total_loss))
        data_history.append(float(MSEU))
        phys_history.append(float(MSEF))
        # Print the loss and the data and physics losses every print_every epochs, Default: 10000 epochs
        if epoch % print_every == 0:
            ratio = float(MSEF) / (float(MSEU) + 1e-12)
            elapsed = time.time() - start_time
            print(
                f"[Adam] Epoch {epoch:5d} | Loss: {float(total_loss):.4e} | "
                f"Data: {float(MSEU):.4e} | PDE: {float(MSEF):.4e} | "
                f"Ratio: {ratio:.2f} | alpha: {alpha:.3f} | t: {elapsed:.1f}s"
            )

        if callback_epoch is not None:
            callback_epoch(epoch, model)

    adam_time = time.time() - start_time
    print(f"\n Adam finalizado en {adam_time:.1f}s")

    return {
        "optimizer": optimizer,
        "loss_history": loss_history,
        "data_history": data_history,
        "phys_history": phys_history,
        "adam_time": adam_time,
    }


def entrenar_lbfgs(
    model,
    X0,
    T0,
    U0_norm,
    X,
    T,
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
):
    # refine the solution obtained with Adam using L-BFGS-B
    print("\n" + "=" * 60)
    print("FASE 3: Refinamiento con L-BFGS")
    print("=" * 60)

    lbfgs_iter = [0]
    lbfgs_losses = []
    lbfgs_start = time.time()

    def callback_lbfgs(flat_weights):
        lbfgs_iter[0] += 1
        if lbfgs_iter[0] % print_every == 0:
            loss_val, _ = loss_and_grads_lbfgs(
                flat_weights,
                model,
                X0,
                T0,
                U0_norm,
                X,
                T,
                k_x,
                k_t,
                c0,
                w_data=w_data,
                w_phys=w_phys,
            )
            lbfgs_losses.append(float(loss_val))
            elapsed = time.time() - lbfgs_start
            print(
                f"[L-BFGS] Iter {lbfgs_iter[0]:4d} | "
                f"Loss: {loss_val:.4e} | t: {elapsed:.1f}s"
            )

    w0 = get_weights(model).astype(np.float64)

    print(f"Pesos iniciales extraídos: {w0.shape[0]} parámetros")
    print("Iniciando L-BFGS...\n")

    result = sopt.minimize(
        fun=loss_and_grads_lbfgs,
        x0=w0,
        args=(model, X0, T0, U0_norm, X, T, k_x, k_t, c0, w_data, w_phys),
        method="L-BFGS-B",
        jac=True,
        callback=callback_lbfgs,
        options={
            "maxiter": maxiter,
            "maxfun": maxfun,
            "ftol": ftol,
            "gtol": gtol,
            "maxls": maxls,
            "disp": False,
        },
    )

    set_weights(model, result.x)

    lbfgs_time = time.time() - lbfgs_start
    print(f"\n L-BFGS finalizado en {lbfgs_time:.1f}s")
    print(f"   Iteraciones        : {result.nit}")
    print(f"   Evaluaciones func  : {result.nfev}")
    print(f"   Loss final         : {result.fun:.4e}")
    print(f"   Convergió          : {result.success}")
    print(f"   Mensaje            : {result.message}")

    return {
        "result": result,
        "lbfgs_iter": lbfgs_iter[0],
        "lbfgs_losses": lbfgs_losses,
        "lbfgs_time": lbfgs_time,
    }
