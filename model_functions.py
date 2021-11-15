import custom_elements as custom
import data_handling as data
import Decoder
import Encoder
import logging
import os
import tensorflow as tf
import utils


logging.basicConfig(filename='logs.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(message)s')


def define_losses(models, input_batch, output_batch):
    losses = {'reconstruction_loss': custom.mse_loss_function(input_batch, output_batch),
              'perceptual_loss': custom.perceptual_loss_function(input_batch, output_batch),
              'kl_divergence_loss': tf.reduce_sum(models['encoder'].losses)}
    losses['total_loss'] = losses['reconstruction_loss'] + losses['perceptual_loss'] + losses['kl_divergence_loss']
    return losses


@tf.function
def train_encoder(models, input_batch, optimizer):
    with tf.GradientTape() as tape:
        latent_space_batch = models['encoder'](input_batch, training=True)
        output_batch = models['decoder'](latent_space_batch, training=False)

        losses = define_losses(models, input_batch, output_batch)

    grads = tape.gradient(losses['total_loss'], models['encoder'].trainable_weights)
    optimizer.apply_gradients(zip(grads, models['encoder'].trainable_weights))
    return losses


@tf.function
def train_decoder(models, input_batch, optimizer):
    with tf.GradientTape() as tape:
        latent_space_batch = models['encoder'](input_batch, training=False)
        output_batch = models['decoder'](latent_space_batch, training=True)

        losses = define_losses(models, input_batch, output_batch)

    grads = tape.gradient(losses['total_loss'], models['decoder'].trainable_weights)
    optimizer.apply_gradients(zip(grads, models['decoder'].trainable_weights))

    return losses


@tf.function
def train_step(models, input_batch, optimizer):
    encoder_losses = train_encoder(models, input_batch, optimizer)
    decoder_losses = train_decoder(models, input_batch, optimizer)
    auto_encoder_losses = utils.setup_loss_dict()
    for loss in auto_encoder_losses:
        auto_encoder_losses[loss] = (encoder_losses[loss] + decoder_losses[loss]) / 2
    return auto_encoder_losses


@tf.function
def validation_step(models, input_batch):
    latent_space_batch = models['encoder'](input_batch, training=False)
    output_batch = models['decoder'](latent_space_batch, training=False)
    losses = define_losses(models, input_batch, output_batch)
    return losses


def train(hyper_parameters, optimizer):
    utils.initialize_directories(hyper_parameters)
    utils.initialize_monitoring_distributions(hyper_parameters)
    models = {'encoder': Encoder.Network(hyper_parameters['latent_dim'], hyper_parameters['beta']),
              'decoder': Decoder.Network()}
    for epoch in range(hyper_parameters['epochs']):
        for step, img_tensor_batch in enumerate(data.train_ds):
            progression = {'epoch': epoch, 'step': step}
            losses = train_step(models, img_tensor_batch, optimizer)
            log_losses(losses, progression, 'TRAINING')
            if (step + 1) % (1 / hyper_parameters['validation_frequency']) == 0:
                validate(models, data.val_ds, 'VALIDATION')
            if (step + 1) % (1 / hyper_parameters['monitoring_frequency']) == 0:
                monitor_progress(progression, models, hyper_parameters)
    save(models, hyper_parameters)


def log_losses(losses, progression, tag):
    training_log = True
    loss_log = ''
    if type(progression) != dict:
        training_log = False
    loss_log += (1 - training_log) * '\n'
    loss_log += f'{tag} - '
    if training_log:
        loss_log += f'Ep: {progression["epoch"] + 1} St: {progression["step"] + 1} - '
    loss_log += f'reconstruction loss = {losses["reconstruction_loss"]:.2f} - '
    loss_log += f'perceptual loss = {losses["perceptual_loss"]:.2f} - '
    loss_log += f'KL divergence loss = {losses["kl_divergence_loss"]:.2f} - '
    loss_log += f'total loss = {losses["total_loss"]:.2f}'
    loss_log += (1 - training_log) * '\n'
    logging.info(loss_log)
    print(loss_log)


def save(models, hyper_parameters):
    models['decoder'].save('Decoder_' + hyper_parameters['name'])
    models['encoder'].save('Encoder_' + hyper_parameters['name'])


def validate(models, dataset, tag):
    iterations = 0
    val_losses = utils.setup_loss_dict()
    for val_step, val_img_tensor in enumerate(dataset):
        iterations += 1
        batch_losses = validation_step(models, val_img_tensor)
        for loss in val_losses:
            val_losses[loss] += batch_losses[loss]
    for loss in val_losses:
        val_losses[loss] /= iterations
    log_losses(val_losses, 'placeholder', tag)


def monitor_progress(progression, models, hyper_parameters):
    monitor_reconstruction(models, progression, hyper_parameters)
    monitor_generation(models, progression, hyper_parameters)


def monitor_reconstruction(models, progression, hyper_parameters):
    path_in = data.directories['monitor_reconstruction']
    img_nr = 0
    for image in os.listdir(path_in):
        img_nr += 1
        input_array = utils.np_array_from_img(path_in, image, hyper_parameters)
        latent_space_representation = models['encoder'](input_array)
        output_tensor = models['decoder'](latent_space_representation)
        img = utils.img_from_tensor(output_tensor)
        loss = int(custom.mse_loss_function(input_array, output_tensor))
        loss += int(custom.perceptual_loss_function(input_array, output_tensor))
        img.save(os.path.join('monitor_reconstruction', f'E{progression["epoch"] + 1}'
                                                        f'S{progression["step"] + 1}'
                                                        f'R{img_nr}'
                                                        f'L{loss}.jpg'))


def monitor_generation(models, progression, hyper_parameters):
    for iteration in range(12):
        output_tensor = models['decoder'](hyper_parameters['normal_distributions'][iteration])
        img = utils.img_from_tensor(output_tensor)
        img.save(os.path.join('monitor_generation', f'E{progression["epoch"] + 1}'
                                                    f'S{progression["step"] + 1}'
                                                    f'G_{iteration + 1}.jpg'))


def test(hyper_parameters):
    models = {'encoder': tf.keras.models.load_model('Encoder_' + hyper_parameters['name']),
              'decoder': tf.keras.models.load_model('Decoder_' + hyper_parameters['name'])}
    validate(models, data.test_ds, 'TEST')


def demo(hyper_parameters):
    for image in range(hyper_parameters['demo_iterations']):
        latent_input = tf.random.normal((1, hyper_parameters['latent_dim']), mean=0, stddev=1)
        saved_model = tf.keras.models.load_model('Decoder_' + hyper_parameters['name'])
        output_tensor = saved_model(latent_input)
        img = utils.img_from_tensor(output_tensor)
        demo_directory = os.path.join(f'demo_{hyper_parameters["name"]}')
        img.save(os.path.join(demo_directory, f'demo_img_{image + 1}.jpg'))
