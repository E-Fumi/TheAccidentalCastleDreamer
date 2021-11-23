import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import regularizers


class SamplingLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(SamplingLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, training=True, **kwargs):
        mean, log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(mean)[0], tf.shape(mean)[1]))
        return mean + tf.exp(0.5 * log_var) * epsilon


class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(filters,
                                  kernel_size,
                                  strides,
                                  padding='same',
                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                  kernel_regularizer=regularizers.l2(0.01))
        self.batch_norm = layers.BatchNormalization()

    def call(self, input_tensor, training=False, **kwargs):
        tensor = self.conv(input_tensor)
        tensor = self.batch_norm(tensor, training=training)
        # tensor = keras.layers.Dropout(0.1)(tensor)
        tensor = keras.activations.relu(tensor)
        return tensor


class EncoderBlock(layers.Layer):
    def __init__(self, filters_1, filters_2, pool_size=2, kernel_size=3, strides=1):
        super(EncoderBlock, self).__init__()
        self.conv_block_1 = CNNBlock(filters_1, kernel_size, strides)
        self.conv_block_2 = CNNBlock(filters_2, kernel_size, strides)
        self.pool_size = pool_size

    def call(self, input_tensor, training=False, **kwargs):
        tensor = self.conv_block_1(input_tensor)
        tensor = self.conv_block_2(tensor)
        tensor = keras.layers.MaxPooling2D(self.pool_size)(tensor)
        return tensor


class DecoderBlock(layers.Layer):
    def __init__(self, filters_1, filters_2, upsampling_factor=2, kernel_size=3, strides=1):
        super(DecoderBlock, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.conv_block_1 = CNNBlock(filters_1, kernel_size, strides)
        self.conv_block_2 = CNNBlock(filters_2, kernel_size, strides)

    def call(self, input_tensor, training=False, **kwargs):
        tensor = layers.UpSampling2D(self.upsampling_factor)(input_tensor)
        tensor = self.conv_block_1(tensor)
        tensor = self.conv_block_2(tensor)
        return tensor


vgg16 = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(None, None, 3))
vgg16.trainable = False

loss_model = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block2_conv2').output)


def mse_loss_function(input_tensor, output_tensor):
    mean_squared_error = tf.math.reduce_sum((input_tensor - output_tensor) ** 2) / input_tensor.shape[0]
    return mean_squared_error


def ae_loss_function(input_tensor, output_tensor):
    absolute_error = tf.math.reduce_sum(abs(input_tensor - output_tensor)) / input_tensor.shape[0]
    return absolute_error


def perceptual_loss_function(input_tensor, output_tensor, gamma):
    d_tensor = abs(loss_model(input_tensor) - loss_model(output_tensor))
    perceptual_loss = tf.math.reduce_sum(d_tensor) / input_tensor.shape[0]
    return perceptual_loss * gamma


def kl_divergence_loss_function(mean, log_var, beta):
    kl_batch = - .5 * k.sum(1 + log_var -
                            k.square(mean) -
                            k.exp(log_var), axis=-1)
    kl_divergence_loss = beta * tf.reduce_mean(kl_batch)
    return kl_divergence_loss
