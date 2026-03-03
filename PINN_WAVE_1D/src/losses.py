import tensorflow as tf


@tf.function
def physics_loss_function(model, X, T, k_x, k_t, c0):
   # Calculate the physics loss
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([X, T])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([X, T])
            u_pred = model(tf.concat([X, T], axis=1))
        u_x = tape1.gradient(u_pred, X)
        u_t = tape1.gradient(u_pred, T)
    u_xx = tape2.gradient(u_x, X)
    u_tt = tape2.gradient(u_t, T)
    del tape1, tape2
    lambda_sq = (c0 * k_x / k_t) ** 2
    F = u_tt - lambda_sq * u_xx
    return tf.reduce_mean(tf.abs(F))


@tf.function
def data_loss_function(model, x_data, t_data, u_data):
    # Calculate the data loss
    u_pred = model(tf.concat([x_data, t_data], axis=1))
    return tf.reduce_mean(tf.square(u_pred - u_data))
