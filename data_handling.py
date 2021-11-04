import tensorflow as tf
import hyperparameters as hp

training_directory = 'C:/Data/buildings_dataset/train'
validation_directory = 'C:/Data/buildings_dataset/val'
test_directory = 'C:/Data/buildings_dataset/test'
monitor_reconstruction_directory = 'C:/Data/buildings_dataset/reconstruct'


def build_dataset(img_directory, shuffle):

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=img_directory,
        label_mode=None,
        batch_size=hp.batch_size,
        image_size=(hp.img_dim, hp.img_dim),
        shuffle=shuffle)

    dataset = dataset.map(lambda data: (data / 255.0))

    return dataset


train_ds = build_dataset(training_directory, True)
val_ds = build_dataset(validation_directory, True)
test_ds = build_dataset(test_directory, False)
