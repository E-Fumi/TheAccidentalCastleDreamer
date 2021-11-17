# Variational Autoencoder

## Intro

This is a [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) (or, more specifically, a [disentangled variational autoencoder](https://arxiv.org/pdf/1812.02833.pdf)), a generative model for the syntheris of architectural imagery. <br/>
The main idea is to have a neural network composed of two convolutional neural networks: an encoder and a decoder. The encoder is meant to encode data into a latent space vector (i.e. an arbitrarily-sized 1D array of float values), and the decoder is meant to reconstruct the original data from that same vector. Once successfully trained, the decoder would ideally be able to construct realistic synthetic data from any vector in the same probability space as those encoded by the encoder for real data. <br/>
<img src="./VAECollage.png" width="548" height="548">

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
tensorflow-gpu<br/>
numpy<br/>
wget<br/>
PIL

## Usage

