from tensorflow import keras

name = 'test_1'
epochs = 20
batch_size = 32
img_dim = 128
latent_dim = 128
beta = 1.0
learning_rate = 0.0001

# TODO:
#  optimizer isn't a hyperparam, it should just be initialized
#  in the main.py and e.g. passed to the training function there
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
# TODO:
#  Also AdamW is often stated as leading to better generalization
#  with convolutional models:
#  https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW

demo_iterations = 100

# TODO:
#  don't use back/forward slashes that will only work on
#  a windows / mac / linux machine. Use https://docs.python.org/3/library/pathlib.html
#  instead (it will by itself recognize the correct separator). Alternatively
#  use https://www.geeksforgeeks.org/python-os-path-join-method/
demo_directory = f'./demo_{name}/'
