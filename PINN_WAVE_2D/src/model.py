import tensorflow as tf


def PINN_WAVE():
    inputs = tf.keras.Input(shape=(3,), name='inputs_xyt')
    x = tf.keras.layers.Dense(128, activation='tanh',
                              kernel_initializer='glorot_normal')(inputs)
    x = tf.keras.layers.Dense(128, activation='tanh',
                              kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dense(128, activation='tanh',
                              kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dense(128, activation='tanh',
                              kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dense(128, activation='tanh',
                              kernel_initializer='glorot_normal')(x)
    outputs = tf.keras.layers.Dense(1, activation=None, name='output_u')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='PINN_WAVE')
