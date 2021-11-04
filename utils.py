import numpy as np
from numpy import asarray
import tensorflow as tf
from tensorflow import keras
import hyperparameters as hp
import encoder
import decoder
import PIL
from PIL import Image
import shutil
import os


def initialize_demo_parameters():
    saved_decoder = keras.models.load_model('Decoder_' + hp.name)
    return hp.demo_iterations, saved_decoder, hp.demo_directory, 'demo_img_'


def np_array_from_img(path, image):
    img_path = os.path.join(path, image)
    img = Image.open(img_path)
    input_img = asarray(img)
    input_img = input_img / 255.0
    return np.array([input_img])


def img_from_tensor(tensor):
    tensor *= 255.0
    tensor = tf.squeeze(tensor)
    array = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(array)


def save(name):
    decoder.network.save('Decoder_' + name)
    encoder.network.save('Encoder_' + name)


def initialize_directories():
    directories = [hp.demo_directory, './monitor_reconstruction/', './monitor_generation/']
    for _ in range(len(directories)):
        if os.path.isdir(directories[_]):
            shutil.rmtree(directories[_])
        os.mkdir(directories[_])
