import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import backend as k
from tensorflow.keras import regularizers
import hyperparameters as hp


class SamplingLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(SamplingLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, training=True, **kwargs):
        mean, log_var = inputs
        kl_batch = - .5 * k.sum(1 + log_var -
                                k.square(mean) -
                                k.exp(log_var), axis=-1)
        self.add_loss(hp.beta * (k.mean(kl_batch)), inputs=inputs)
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(mean)[0], tf.shape(mean)[1]))
        return mean + tf.exp(0.5 * log_var) * epsilon


class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(filters,
                                  kernel_size,
                                  strides,
                                  padding='same',
                                  kernel_regularizer=regularizers.l2(0.01))
        self.batch_norm = layers.BatchNormalization()

    def call(self, input_tensor, training=False, **kwargs):
        tensor = self.conv(input_tensor)
        tensor = self.batch_norm(tensor, training=training)
        tensor = keras.activations.relu(tensor)
        return tensor


e_netB3 = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=(None, None, 3))
e_netB3.trainable = False

loss_model = Model(inputs=e_netB3.input, outputs=e_netB3.get_layer('top_conv').output)


def loss_function(input_tensor, output_tensor):
    custom_loss = tf.math.reduce_sum(abs(loss_model(input_tensor) - loss_model(output_tensor))) / input_tensor.shape[0]
    custom_loss += tf.math.reduce_sum((input_tensor - output_tensor) ** 2) / input_tensor.shape[0]
    return custom_loss


def mse_loss_function(input_tensor, output_tensor):
    mean_squared_error = tf.math.reduce_sum((input_tensor - output_tensor) ** 2) / input_tensor.shape[0]
    return mean_squared_error


def ae_loss_function(input_tensor, output_tensor):
    absolute_error = tf.math.reduce_sum(abs(input_tensor - output_tensor)) / input_tensor.shape[0]
    return absolute_error


def perceptual_loss_function(input_tensor, output_tensor):
    d_tensor = abs(loss_model(input_tensor) - loss_model(output_tensor))
    perceptual_loss = tf.math.reduce_sum(d_tensor) / input_tensor.shape[0]
    return perceptual_loss
