import model_functions as model
import utils
import deployment

if __name__ == '__main__':
    model.train()
    utils.fetch_images()
    deployment.sort_images()
