import os

import numpy as np
import tensorflow as tf

import custom_elements as custom
import data_handling as data
import decoder
import encoder
import hyperparameters as hp
import utils


@tf.function
def train_encoder(input_batch, optimizer):
    with tf.GradientTape() as tape:
        latent_space_batch = encoder.network(input_batch, training=True)
        output_batch = decoder.network(latent_space_batch, training=False)

        reconstruction_loss = custom.loss_function(input_batch, output_batch)
        kl_divergence_loss = tf.reduce_sum(encoder.network.losses)
        total_loss = reconstruction_loss + kl_divergence_loss

    grads = tape.gradient(total_loss, encoder.network.trainable_weights)
    optimizer.apply_gradients(zip(grads, encoder.network.trainable_weights))
    return total_loss, reconstruction_loss, kl_divergence_loss


@tf.function
def train_decoder(input_batch, optimizer):
    with tf.GradientTape() as tape:
        latent_space_batch = encoder.network(input_batch, training=False)
        output_batch = decoder.network(latent_space_batch, training=True)

        reconstruction_loss = custom.loss_function(input_batch, output_batch)
        kl_divergence_loss = tf.reduce_sum(encoder.network.losses)
        total_loss = reconstruction_loss + kl_divergence_loss

    grads = tape.gradient(total_loss, decoder.network.trainable_weights)
    optimizer.apply_gradients(zip(grads, decoder.network.trainable_weights))
    return total_loss, reconstruction_loss, kl_divergence_loss


@tf.function
def validation_step(input_batch):
    latent_space_batch = encoder.network(input_batch, training=False)
    output_batch = decoder.network(latent_space_batch, training=False)

    reconstruction_loss = custom.loss_function(input_batch, output_batch)
    kl_divergence_loss = tf.reduce_sum(encoder.network.losses)
    total_loss = reconstruction_loss + kl_divergence_loss

    return total_loss, reconstruction_loss, kl_divergence_loss


@tf.function
def train_step(input_batch, optimizer):
    encoder_training_losses = train_encoder(input_batch, optimizer)
    decoder_training_losses = train_decoder(input_batch, optimizer)
    total_loss = (encoder_training_losses[0] + decoder_training_losses[0]) / 2
    reconstruction_loss = (encoder_training_losses[1] + decoder_training_losses[
        1]) / 2
    kl_divergence_loss = (encoder_training_losses[2] + decoder_training_losses[
        2]) / 2
    return total_loss, reconstruction_loss, kl_divergence_loss


def train(nr_of_epochs, optimizer):
    utils.initialize_directories()
    for epoch in range(nr_of_epochs):
        for step, img_tensor_batch in enumerate(data.train_ds):
            loss, reconstruction_loss, kl_loss = train_step(img_tensor_batch,
                                                            optimizer)
            # TODO:
            #  use the logging module wherever you call print()
            #  https://docs.python.org/3/library/logging.html
            print(f'Ep: {epoch + 1} St: {step + 1} - '
                  f'reconstruction loss = {reconstruction_loss:.2f} - '
                  f'KL divergence loss = {kl_loss:.2f} - '
                  f'total loss = {loss:.2f}')

            # TODO:
            #  good practice is is to let the number of steps
            #  between validation / progress logging be a parameter
            if (step + 1) % 100 == 0:
                validate(data.val_ds, 'VALIDATION')
            if (step + 1) % 1000 == 0:
                monitor_progress(epoch, step)


def validate(dataset, tag):
    iterations = 0
    val_losses = np.array([0., 0., 0.])
    for val_step, val_img_tensor in enumerate(dataset):
        iterations += 1
        val_loss, val_mse_loss, val_kl_loss = validation_step(val_img_tensor)
        batch_losses = np.array([val_loss, val_mse_loss, val_kl_loss])
        val_losses += batch_losses
    val_losses /= iterations
    print(f'\n{tag} - '
          f'reconstruction loss = {val_losses[1]:.2f} - '
          f'KL divergence loss = {val_losses[2]:.2f} - '
          f'total loss = {val_losses[0]:.2f}\n')


def monitor_progress(epoch, step):
    monitor_reconstruction(epoch, step)
    monitor_generation(epoch, step)


def monitor_reconstruction(epoch, step):
    path_in = data.monitor_reconstruction_directory
    img_nr = 1
    for image in os.listdir(path_in):
        input_array = utils.np_array_from_img(path_in, image)
        latent_space_representation = encoder.network(input_array)
        output_tensor = decoder.network(latent_space_representation)
        img = utils.img_from_tensor(output_tensor)
        loss = int(custom.loss_function(input_array, output_tensor))
        img.save(
            f'./monitor_reconstruction/E{epoch + 1}S{step + 1}R{img_nr}L{loss}.jpg')
        img_nr += 1


def monitor_generation(epoch, step):
    demo((12, decoder.network, './monitor_generation/',
          f'E{epoch + 1}S{step + 1}G_'))


def test():
    validate(data.test_ds, 'TEST')


def demo(demo_parameters):
    # TODO:
    #  consider using a dictionary instead of a list to unpack
    #  those parameters, makes it more readable than indexing by number
    iterations = demo_parameters[0]
    network = demo_parameters[1]
    demo_directory = demo_parameters[2]
    file_prefix = demo_parameters[3]
    for image in range(iterations):
        latent_input = tf.random.normal((1, hp.latent_dim), mean=0, stddev=1)
        output_tensor = network(latent_input)
        img = utils.img_from_tensor(output_tensor)
        img.save(demo_directory + f'{file_prefix}{image + 1}.jpg')
