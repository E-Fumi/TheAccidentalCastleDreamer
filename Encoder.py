import custom_elements as custom
from tensorflow import keras
from tensorflow.keras import layers


class Network(keras.Model):
    def __init__(self, latent_space_dim):
        super(Network, self).__init__()
        self.Block_1 = custom.EncoderBlock(64, 64)
        self.Block_2 = custom.EncoderBlock(64, 64)
        self.Block_3 = custom.EncoderBlock(128, 128)
        self.Block_4 = custom.EncoderBlock(128, 128)
        self.Block_5 = custom.EncoderBlock(256, 256)
        self.Block_6 = custom.EncoderBlock(256, 256)
        self.Dense = layers.Dense(512, activation="relu")
        self.Mean = layers.Dense(latent_space_dim, activation="relu", name="z_mean")
        self.Log_Var = layers.Dense(latent_space_dim, activation="relu", name="z_log_var")
        self.Sampling_Layer = custom.SamplingLayer()

    def call(self, input_tensor, training=False):
        x = self.Block_1(input_tensor, training=training)
        x = self.Block_2(x, training=training)
        x = self.Block_3(x, training=training)
        x = self.Block_4(x, training=training)
        x = self.Block_5(x, training=training)
        x = self.Block_6(x, training=training)
        x = layers.Flatten()(x)
        x = self.Dense(x, training=training)
        z_mean = self.Mean(x, training=training)
        z_log_var = self.Log_Var(x, training=training)
        latent_space_encoding = self.Sampling_Layer([z_mean, z_log_var])
        return latent_space_encoding, z_mean, z_log_var
