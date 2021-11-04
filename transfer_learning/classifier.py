from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

DenseNet_base = keras.applications.DenseNet201(
    include_top=False,
    weights='imagenet',
    input_shape=(300, 300, 3))

DenseNet = keras.Sequential([
    keras.Input(shape=(300, 300, 3)),
    DenseNet_base,
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    layers.Dense(1, activation='sigmoid')
])
