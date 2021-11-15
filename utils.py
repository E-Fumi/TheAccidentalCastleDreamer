import numpy as np
import os
import PIL
import shutil
import sys
import tensorflow as tf
import wget


datasets = {'validation': 'https://zenodo.org/record/5654571/files/validation.zip?download=1',
            'test': 'https://zenodo.org/record/5655436/files/test.zip?download=1',
            'reduced_128': 'https://zenodo.org/record/5655491/files/reduced_128.zip?download=1',
            'reduced_256_part_1': 'https://zenodo.org/record/5676514/files/red_256_part_1.zip.zip?download=1',
            'reduced_256_part_2': 'https://zenodo.org/record/5676514/files/red_256_part_2.zip.zip?download=1',
            'reduced_256_part_3': 'https://zenodo.org/record/5676514/files/red_256_part_3.zip.zip?download=1',
            'reduced_256_part_4': 'https://zenodo.org/record/5676514/files/red_256_part_4.zip.zip?download=1',
            'reduced_256_part_5': 'https://zenodo.org/record/5676514/files/red_256_part_5.zip.zip?download=1'}


def progress_bar(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def fetch_dataset(dataset, msg):
    if os.path.isdir(os.path.join(dataset)):
        pass
    else:
        print(msg)
        download_and_extract(dataset)


def download_and_extract(dataset):
    wget.download(datasets[dataset], bar=progress_bar)
    os.mkdir(os.path.join(dataset))
    print('\nextracting...')
    shutil.unpack_archive(dataset + '.zip', os.path.join(dataset))
    os.remove(dataset + '.zip')
    print('')


def fetch_datasets(hyper_parameters):
    directories = {'validation': os.path.join('validation'),
                   'test': os.path.join('test')}
    if hyper_parameters['img_dim'] < 129:
        fetch_dataset('reduced_128', 'Training Dataset')
        directories['train'] = os.path.join('reduced_128')
    else:
        fetch_reduced_256_dataset()
        directories['train'] = os.path.join('reduced_256')
    fetch_dataset('validation', 'Validation Dataset')
    fetch_dataset('test', 'Test Dataset')
    return directories


def fetch_reduced_256_dataset():
    if os.path.isdir(os.path.join('reduced_256')):
        pass
    else:
        print('Training Dataset\n(comes in 5 ~1GB chunks, now might be a good time for a coffee)')
        tmp_dir = os.path.join('reduced_256_tmp')
        os.mkdir(tmp_dir)
        for part in range(5):
            print(f'\npart {part + 1} / 5')
            wget.download(datasets[f'reduced_256_part_{part + 1}'], bar=progress_bar)
            os.rename(f'red_256_part_{part + 1}.zip.zip', 'reduced_256.zip')
            print('\nextracting...')
            shutil.unpack_archive(f'reduced_256.zip', tmp_dir)
            os.rename(os.path.join(tmp_dir, f'red_256_part_{part + 1}'),
                      os.path.join(tmp_dir, f'red_256_part_{part + 2}'))
            os.remove(f'reduced_256.zip')
        os.rename(os.path.join(tmp_dir, 'red_256_part_6'), os.path.join(tmp_dir, 'images'))
        os.rename(tmp_dir, os.path.join('reduced_256'))
        print('')


def setup_loss_dict():
    dictionary = {'total_loss': 0.,
                  'reconstruction_loss': 0.,
                  'perceptual_loss': 0.,
                  'kl_divergence_loss': 0.}
    return dictionary


def np_array_from_img(path, image, hyper_parameters):
    img_path = os.path.join(path, image)
    img = PIL.Image.open(img_path)
    resized_img = img.resize((hyper_parameters['img_dim'], hyper_parameters['img_dim']), PIL.Image.ANTIALIAS)
    input_img = np.asarray(resized_img)
    input_img = input_img / 255.0
    return np.array([input_img])


def img_from_tensor(tensor):
    tensor *= 255.0
    tensor = tf.squeeze(tensor)
    array = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(array)


def initialize_monitoring_distributions(hyper_parameters):
    random_distributions = []
    for _ in range(12):
        random_distribution = tf.random.normal((1, hyper_parameters['latent_dim']), mean=0, stddev=1)
        random_distributions.append(random_distribution)
    hyper_parameters['normal_distributions'] = random_distributions


def initialize_directories(hyper_parameters):
    directories = [os.path.join(f'demo_{hyper_parameters["name"]}'),
                   os.path.join('monitor_reconstruction'),
                   os.path.join('monitor_generation')]
    for _ in range(len(directories)):
        if os.path.isdir(directories[_]):
            shutil.rmtree(directories[_])
        os.mkdir(directories[_])
