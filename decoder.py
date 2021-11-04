from tensorflow import keras
from tensorflow.keras import layers
import hyperparameters as hp
import custom_elements as custom


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
