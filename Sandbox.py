import tensorflow as tf
from tensorflow.keras import Model
import data_handling as data

e_netB3 = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=(None, None, 3))
e_netB3.trainable = False

loss_model = Model(inputs=e_netB3.input, outputs=e_netB3.get_layer('top_conv').output)

for step, img_tensor_batch in enumerate(data.test_ds):
    print(e_netB3(img_tensor_batch))
