# The Accidental Castle Dreamer

## Intro
This generative model is a [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) (or, more specifically, a [disentangled variational autoencoder](https://arxiv.org/pdf/1812.02833.pdf)) meant for the synthesis of architectural imagery. The data it works with is basically a large aggregation of vacation pictures, and it turns out that people are much more likely to take pictures of castles and churches than of regular office buildings, which in turn skews the model's reconstruction's probability landscape, hence the name of the project. <br/>

The main idea is to have a neural network composed of two convolutional neural networks: an encoder and a decoder. The encoder is meant to encode data into a latent space vector (i.e. an arbitrarily-sized 1D array of float values), and the decoder is meant to reconstruct the original data from that same vector. Once successfully trained, the decoder would ideally be able to construct realistic synthetic data from any set of values in the same probability space as the latent space vectors encoded by the encoder for real data.<br/>
<br/>
<p align="center">
  <img src="./VAECollage.png" width="548" height="548"><br/>
  Each of these (admittedly somewhat cherrypicked)<br/>
  images comes from a random normal distribution.
 </p>
<br/>
variational aspect<br/>
disentanglement<br/>

## Network Details

First things first, this is a work in progress, and while the base of it is (passably) solid, various details are going to change in the coming weeks, hopefully bringing with them improvements in performance.

### Data Preparation

All data used to train this network stems from the Google Landmarks Dataset v2, a collection of 5 million pictures of human-made and natural landmarks.<br/>

Images depicting architecture were selected from the original dataset by two ensemble classifiers. The first model discerns whether a picture is of a building or not, and the second discerns whether an image of a building is suitable or not (this latter task is admittedly vague and based on a set of arbitrary criteria such as whether a photo is blurry, whether a significant portion of the architecture is blocked by a vehicle, or whether the photo contains enough features of a building to infer its overall structure).<br/>

Each ensemble model is composed of three identical networks that were trained independently of one another. The architecture of the individual networks is always an imagenet-pretrained Densenet201's convolutional base with three dense layers added at the end, trained on a smaller, painstakingly hand-annotated dataset (approximately 12000 pictures in total). The smaller datasets were fed to a keras image data generator for data augmentation.<br/>

All pertinent scripts are in the 'transfer_learning' folder.<br/>

### Losses
The overall loss function is composed of 3 independent components:<br/>
- a simple reconstruction loss equal to the mean squared error between all corresponding values of the input and output tensors.<br/>
- a KL-divergence loss function to incentivise the latent space distribution to occupy the same space. <br/>
- a perceptual loss function (which I have only recently began tinkering with) hinging on a pre-trained network.<br/>
### Architecture
Placeholder text.
### Monitoring
Placeholder text.
### Hardware
Placeholder text.

## Requirements
python 3.x<br/>
[Conda](https://docs.conda.io/en/latest/miniconda.html) (not strictly a requirement, but is likely to make running this project a great deal easier)<br/>
Cuda drivers (necessary if you want to run this code on a GPU)<br/>
tensorflow-gpu (again, necessary to run this on a GPU, otherwise regular tensorflow will do)<br/>
numpy<br/>
wget<br/>
PIL

## Usage
With Conda:<br/>
`conda install -c anaconda cudatoolkit`<br/>
`git clone https://github.com/E-Fumi/TheAccidentalCastleDreamer`<br/>
`cd TheAccidentalCastleDreamer`<br/>
`conda env create -f environment.yml`<br/>
`conda activate environment`<br/>
`python main.py`<br/>

Without Conda:<br/>
`git clone https://github.com/E-Fumi/VariationalAutoEncoder`<br/>
`cd VariationalAutoEncoder`<br/>
`pip install -r requirements.txt`<br/>
`python main.py`<br/>

