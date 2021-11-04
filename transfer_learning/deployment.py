import numpy as np
import os
import shutil
from tqdm import tqdm
from tensorflow import keras
import tensorflow as tf

M1 = keras.models.load_model('FirstPass_D_1')
M2 = keras.models.load_model('FirstPass_D_2')
M3 = keras.models.load_model('FirstPass_D_3')
N1 = keras.models.load_model('SecondPass_D_1')
N2 = keras.models.load_model('SecondPass_D_2')
N3 = keras.models.load_model('SecondPass_D_3')

directory = "C://Data/landmarks_dataset/0"
destination = "C://Data/landmarks_dataset/1"


def sort_images():
    for file in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, file)
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(300, 300))
        input_array = tf.keras.preprocessing.image.img_to_array(image)
        input_array = input_array / 255.0
        input_array = np.array([input_array])
        first_pass(input_array, img_path, file)


def first_pass(input_array, img_path, file):
    if M1.predict(input_array) > 0.9 and M2.predict(input_array) > 0.9 and M3.predict(input_array) > 0.9:
        second_pass(input_array, img_path, file)
    else:
        os.remove(img_path)


def second_pass(input_array, img_path, file):
    if N1.predict(input_array) > 0.9 and N2.predict(input_array) > 0.9 and N3.predict(input_array) > 0.9:
        shutil.move(img_path, os.path.join(destination, file))
    else:
        os.remove(img_path)
