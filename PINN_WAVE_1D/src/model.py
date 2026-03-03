import tensorflow as tf


def PINN_WAVE(layers=5, neurons=128, activation="softplus"):
    # Build a 1D PINN model, input=2(x,t), output=1 (Ez)
    inputs = tf.keras.Input(shape=(2,), name="inputs_xt")
    x = tf.keras.layers.Dense(
        neurons,
        activation=activation,
        kernel_initializer="glorot_normal",
    )(inputs)

    for _ in range(layers - 1):
        x = tf.keras.layers.Dense(
            neurons,
            activation=activation,
            kernel_initializer="glorot_normal",
        )(x)

    outputs = tf.keras.layers.Dense(1, activation=None, name="output_u")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="PINN_WAVE")
