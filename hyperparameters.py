from tensorflow import keras

name = 'test_1'
epochs = 20
batch_size = 32
img_dim = 128
latent_dim = 128
beta = 1.0
learning_rate = 0.0001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

demo_iterations = 100
demo_directory = f'./demo_{name}/'
