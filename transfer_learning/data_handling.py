import hyperparameters as hp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_augmenter = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    rotation_range=25,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    shear_range=0.1
)


def build_generator(augmenter, directory):
    generator = augmenter.flow_from_directory(
        directory='C://Data/2nd_pass/' + directory,
        target_size=(300, 300),
        color_mode='rgb',
        batch_size=hp.batch_size,
        class_mode='binary',
        shuffle=True)
    return generator


training_generator = build_generator(data_augmenter, 'train')


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='C://Data/2nd_pass/validation',
        labels='inferred',
        label_mode='binary',
        batch_size=32,
        image_size=(300, 300),
        shuffle=True)

val_ds = val_ds.map(lambda a, b: (a / 255.0, tf.cast(b, dtype=tf.int8)))