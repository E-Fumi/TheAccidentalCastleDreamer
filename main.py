import hyperparameters as hp
import model_functions as model
import utils

if __name__ == '__main__':
    # TODO:
    #  make hyperparameters be passed to the train function
    #  via https://docs.python.org/3/library/argparse.html
    #  with defaults specified here, but capable of being
    #  over-written via the command line arguments
    #  (this is also shown in the Set Transfomer repo I linked via fb)
    model.train(hp.epochs, hp.optimizer)
    utils.save(hp.name)
    model.test()
    model.demo(utils.initialize_demo_parameters())
