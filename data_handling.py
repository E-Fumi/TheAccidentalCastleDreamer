import hyperparameters as hyper
import os
import tensorflow as tf
import utils


directories = utils.fetch_datasets(hyper.parameters)
directories['monitor_reconstruction'] = os.path.join('to_reconstruct')


def build_dataset(img_directory, shuffle):

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=img_directory,
        label_mode=None,
        batch_size=hyper.parameters['batch_size'],
        image_size=(hyper.parameters['img_dim'], hyper.parameters['img_dim']),
        shuffle=shuffle)

    dataset = dataset.map(lambda data: (data / 255.0))

    return dataset


train_ds = build_dataset(directories['train'], True)
val_ds = build_dataset(directories['validation'], True)
test_ds = build_dataset(directories['test'], False)
