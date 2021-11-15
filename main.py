from tensorflow import keras
import hyperparameters as hyper
import model_functions as model


optimizer = keras.optimizers.Adam(learning_rate=hyper.parameters['learning_rate'])

if __name__ == '__main__':
    model.train(hyper.parameters, optimizer)
    model.test(hyper.parameters)
    model.demo(hyper.parameters)
