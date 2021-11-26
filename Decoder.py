import custom_elements as custom
from tensorflow import keras
from tensorflow.keras import layers


class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Dense = layers.Dense(1024, activation="relu")
        self.Reshape = layers.Reshape((4, 4, 64))
        self.Block_1 = custom.DecoderBlock(512, 512)
        self.Block_2 = custom.DecoderBlock(256, 256)
        self.Block_3 = custom.DecoderBlock(128, 128)
        self.Block_4 = custom.DecoderBlock(64, 64)
        self.Block_5 = custom.DecoderBlock(64, 64)
        self.Last_Conv = layers.Conv2D(3, 3, activation="sigmoid", padding="same")

    def call(self, input_tensor, training=False):
        x = self.Dense(input_tensor, training=training)
        x = self.Reshape(x)
        x = self.Block_1(x, training=training)
        x = self.Block_2(x, training=training)
        x = self.Block_3(x, training=training)
        x = self.Block_4(x, training=training)
        x = self.Block_5(x, training=training)
        x = self.Last_Conv(x)
        return x
