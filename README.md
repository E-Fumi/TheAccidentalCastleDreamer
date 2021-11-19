# The Accidental Castle Dreamer

## Intro

This generative model is a [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) (or, more specifically, a [disentangled variational autoencoder](https://arxiv.org/pdf/1812.02833.pdf)) meant for the synthesis of architectural imagery. The data it works with is basically a large aggregation of vacation pictures, and I discovered that people are much more likely to take pictures of castles and churches than of regular office buildings, which in turn skews the model's reconstruction's probability landscape, hence the name.<br/>

The main idea is to have a neural network composed of two convolutional neural networks: an encoder and a decoder. The encoder is meant to encode data into a latent space vector (i.e. an arbitrarily-sized 1D array of float values), and the decoder is meant to reconstruct the original data from that same vector. Once successfully trained, the decoder would ideally be able to construct realistic synthetic data from any vector in the same probability space as those encoded by the encoder for real data. <br/>
<p align="center">
  <img src="./VAECollage.png" width="548" height="548">
 </p>

## Network Details

First things first, this is a work in progress, and while the base of it is (passably) solid, various details are going to change in the coming weeks, hopefully bringing with them improvements in performance.

### Data Preparation
Placeholder text.
### Losses
Placeholder text.
### Architecture
Placeholder text.
### Monitoring
Placeholder text.
### Hardware
Placeholder text.

## Requirements
python 3.x<br/>
Conda (not strictly a requirement, but is likely to make running this project a great deal easier)<br/>
Cuda drivers (necessary if you want to run this code on a GPU)<br/>
tensorflow-gpu (again, necessary to run this on a GPU, otherwise regular tensorflow will do)<br/>
numpy<br/>
wget<br/>
PIL

## Usage
`git clone https://github.com/E-Fumi/VariationalAutoEncoder`<br/>
`cd VariationalAutoEncoder`<br/>
`pip install -r requirements.txt`<br/>
`python main.py`<br/>

