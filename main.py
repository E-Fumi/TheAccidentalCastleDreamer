import hyperparameters as hp
import model_functions as model
import utils

if __name__ == '__main__':
    model.train(hp.epochs, hp.optimizer)
    utils.save(hp.name)
    model.test()
    model.demo(utils.initialize_demo_parameters())
