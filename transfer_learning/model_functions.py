from tensorflow import keras
import data_handling as data
import classifier
import hyperparameters as hp


class CustomCallback(keras.callbacks.Callback):
    def on_test_end(self, logs=None):
        if logs.get('accuracy') > 0.91:
            print('Saving!')
            classifier.DenseNet.save('SecondPass_D_3')
            self.model.stop_training = True


def unfreeze_layers(increment):
    classifier.DenseNet_base.trainable = True
    set_trainable = False
    for layer in classifier.DenseNet_base.layers:
        if layer.name[:14 - increment] == 'conv5_block32_'[:-increment]:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False


def train():
    for stage in range(hp.stages):
        if stage == 0:
            classifier.DenseNet_base.trainable = False
        else:
            unfreeze_layers(stage)

        classifier.DenseNet.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.RMSprop(learning_rate=hp.learning_rate),
            metrics=['accuracy'])

        for loop in range(hp.epochs // hp.stages):
            classifier.DenseNet.fit(data.training_generator, steps_per_epoch=150)
            classifier.DenseNet.evaluate(data.val_ds, callbacks=[CustomCallback()])
            print(f'concluded epoch {stage * (hp.epochs // hp.stages) + loop + 1} / {hp.epochs}')
