import os
import shutil

import PIL
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy import asarray
from tensorflow import keras

import decoder
import encoder
import hyperparameters as hp


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
    # TODO:
    #  possibly consider generating a run id and adding
    #  it to the name. Particularly useful when your logger
    #  also saves its output to a file and then you can identify
    #  both the logs and the model by the same id of the run of main.py
    decoder.network.save('Decoder_' + name)
    encoder.network.save('Encoder_' + name)


def initialize_directories():
    directories = [hp.demo_directory, './monitor_reconstruction/',
                   './monitor_generation/']
    for _ in range(len(directories)):
        if os.path.isdir(directories[_]):
            shutil.rmtree(directories[_])
        os.mkdir(directories[_])
