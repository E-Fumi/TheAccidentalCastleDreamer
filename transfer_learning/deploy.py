import numpy as np
import os
import shutil
from tensorflow import keras
import tensorflow as tf
import sys
import wget

origin = "path/to/your/unsorted/images"
destination = "path/to/where/you/want/the/sorted/ones/to/go"

models = {'FirstPass_D_1': 'https://zenodo.org/record/5715671/files/FirstPass_D_1.zip?download=1',
          'FirstPass_D_2': 'https://zenodo.org/record/5715671/files/FirstPass_D_2.zip?download=1',
          'FirstPass_D_3': 'https://zenodo.org/record/5715671/files/FirstPass_D_3.zip?download=1',
          'SecondPass_D_1': 'https://zenodo.org/record/5715671/files/SecondPass_D_1.zip?download=1',
          'SecondPass_D_2': 'https://zenodo.org/record/5715671/files/SecondPass_D_2.zip?download=1',
          'SecondPass_D_3': 'https://zenodo.org/record/5715671/files/SecondPass_D_3.zip?download=1'}


def fetch_models(model_dictionary):
    model_nr = 0
    for model in model_dictionary:
        if os.path.isdir(os.path.join(model)):
            pass
        else:
            model_nr += 1
            print(f'Downloading {model} - {model_nr}/6')
            wget.download(model_dictionary[model], bar=progress_bar)
            print('\nextracting...')
            shutil.unpack_archive(model + '.zip', os.path.join(model))
            os.remove(model + '.zip')
            print('')


def progress_bar(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def sort_images(path_in, path_out):
    index = 0
    nr_of_files = len(os.listdir(path_in))
    classifiers = load_models()
    for file in os.listdir(path_in):
        index += 1
        args = {'path_in': path_in,
                'path_out': path_out,
                'file': file,
                'image_path': os.path.join(path_in, file)}
        image = tf.keras.preprocessing.image.load_img(args['image_path'], target_size=(300, 300))
        input_array = prepare_array(image)
        first_pass(input_array, classifiers, args)
        print(f'sorted {index} of {nr_of_files} images')


def prepare_array(img):
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = array / 255.0
    array = np.array([array])
    return array


def first_pass(input_array, c, args):
    if c['F1'].predict(input_array) > 0.9 and c['F2'].predict(input_array) > 0.9 and c['F3'].predict(input_array) > 0.9:
        second_pass(input_array, c, args)
    else:
        pass


def second_pass(input_array, c, args):
    if c['S1'].predict(input_array) > 0.9 and c['S2'].predict(input_array) > 0.9 and c['S3'].predict(input_array) > 0.9:
        shutil.move(args['image_path'], os.path.join(args['path_out'], args['file']))
    else:
        pass


def load_models():
    print('The models may take a few minutes to load.')
    classifiers = {}
    passes = ['First', 'Second']
    for model in range(6):
        print(f'Loading model {model + 1} / 6')
        classifier = keras.models.load_model(f'{passes[model // 3]}Pass_D_{(model % 3) + 1}')
        classifiers[f'{passes[model // 3][0]}{(model % 3) + 1}'] = classifier
    return classifiers


fetch_models(models)
sort_images(origin, destination)
