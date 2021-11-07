from tensorflow import keras
from tensorflow.keras import layers

import custom_elements as custom
import hyperparameters as hp

# TODO:
#  I would add some Dropout!
#  https://keras.io/api/layers/regularization_layers/dropout/
inputs = keras.Input(shape=(hp.img_dim, hp.img_dim, 3))

x = custom.CNNBlock(64)(inputs)
x = custom.CNNBlock(64)(x)
x = layers.MaxPooling2D(2)(x)

x = custom.CNNBlock(64)(x)
x = custom.CNNBlock(64)(x)
x = layers.MaxPooling2D(2)(x)

x = custom.CNNBlock(128)(x)
x = custom.CNNBlock(128)(x)
x = layers.MaxPooling2D(2)(x)

x = custom.CNNBlock(128)(x)
x = custom.CNNBlock(128)(x)
x = layers.MaxPooling2D(2)(x)

x = custom.CNNBlock(256)(x)
x = custom.CNNBlock(256)(x)
x = layers.MaxPooling2D(2)(x)

x = custom.CNNBlock(256)(x)
x = custom.CNNBlock(256)(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)

x = layers.Dense(512, activation="relu")(x)

z_mean = layers.Dense(hp.latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(hp.latent_dim, name="z_log_var")(x)
latent_space_encoding = custom.SamplingLayer()([z_mean, z_log_var])

network = keras.Model(inputs, latent_space_encoding, name="encoder")
