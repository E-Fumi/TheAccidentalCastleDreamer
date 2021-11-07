from tensorflow import keras
from tensorflow.keras import layers

import custom_elements as custom
import hyperparameters as hp

# TODO:
#  create a separate Decoder() and Encoder() class
#  that inherits from keras.Model. The one thing about
#  your code that jumps out is not using abstraction.
#  more info on how to do that here: https://www.tensorflow.org/guide/keras/custom_layers_and_models
latent_space_encoding = keras.Input(shape=(hp.latent_dim,))
x = layers.Dense(1024, activation="relu")(latent_space_encoding)
x = layers.Reshape((4, 4, 64))(x)

x = layers.UpSampling2D(2)(x)
x = custom.CNNBlock(512)(x)
x = custom.CNNBlock(512)(x)
# x = custom.CNNBlock(512)(x)

x = layers.UpSampling2D(2)(x)
x = custom.CNNBlock(256)(x)
x = custom.CNNBlock(256)(x)
# x = custom.CNNBlock(256)(x)

x = layers.UpSampling2D(2)(x)
x = custom.CNNBlock(128)(x)
x = custom.CNNBlock(128)(x)
# x = custom.CNNBlock(128)(x)

x = layers.UpSampling2D(2)(x)
x = custom.CNNBlock(64)(x)
x = custom.CNNBlock(64)(x)
# x = custom.CNNBlock(64)(x)

x = layers.UpSampling2D(2)(x)
x = custom.CNNBlock(64)(x)
x = custom.CNNBlock(64)(x)
# x = custom.CNNBlock(64)(x)

decoder_outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

network = keras.Model(latent_space_encoding, decoder_outputs, name="decoder")
