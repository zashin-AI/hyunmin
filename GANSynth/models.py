import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import metrics
import spectral_ops


class GANSynth(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, spectral_params, hyper_params):

        # -----------------------------------------------------------------------------------------
        # Non-Saturating Loss + Mode-Seeking Loss + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis]
        # (https://arxiv.org/pdf/1903.05628.pdf)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------

        real_waveforms, labels = real_input_fn()
        latents = fake_input_fn()

        fake_images = generator(latents, labels)

        real_magnitude_spectrograms, real_instantaneous_frequencies = spectral_ops.convert_to_spectrogram(real_waveforms, **spectral_params)
        real_images = tf.stack([real_magnitude_spectrograms, real_instantaneous_frequencies], axis=1)

        fake_magnitude_spectrograms, fake_instantaneous_frequencies = tf.unstack(fake_images, axis=1)
        fake_waveforms = spectral_ops.convert_to_waveform(fake_magnitude_spectrograms, fake_instantaneous_frequencies, **spectral_params)

        real_features, real_logits = discriminator(real_images, labels)
        fake_features, fake_logits = discriminator(fake_images, labels)

        # label conditioning from
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        real_logits = tf.gather_nd(real_logits, indices=tf.where(labels))
        fake_logits = tf.gather_nd(fake_logits, indices=tf.where(labels))

        # non-saturating loss
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_gradient_penalty_weight:
            real_gradients = tf.gradients(real_logits, [real_images])[0]
            real_gradient_penalties = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])
            discriminator_losses += real_gradient_penalties * hyper_params.real_gradient_penalty_weight
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_gradient_penalty_weight:
            fake_gradients = tf.gradients(fake_logits, [fake_images])[0]
            fake_gradient_penalties = tf.reduce_sum(tf.square(fake_gradients), axis=[1, 2, 3])
            discriminator_losses += fake_gradient_penalties * hyper_params.fake_gradient_penalty_weight

        # non-saturating loss
        generator_losses = tf.nn.softplus(-fake_logits)
        # gradient-based mode-seeking loss
        if hyper_params.mode_seeking_loss_weight:
            latent_gradients = tf.gradients(fake_images, [latents])[0]
            mode_seeking_losses = 1.0 / (tf.reduce_sum(tf.square(latent_gradients), axis=[1]) + 1.0e-6)
            generator_losses += mode_seeking_losses * hyper_params.mode_seeking_loss_weight

        generator_loss = tf.reduce_mean(generator_losses)
        discriminator_loss = tf.reduce_mean(discriminator_losses)

        generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.generator_learning_rate,
            beta1=hyper_params.generator_beta1,
            beta2=hyper_params.generator_beta2
        )
        discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.discriminator_learning_rate,
            beta1=hyper_params.discriminator_beta1,
            beta2=hyper_params.discriminator_beta2
        )

        generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        generator_train_op = generator_optimizer.minimize(
            loss=generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )

        self.real_waveforms = real_waveforms
        self.fake_waveforms = fake_waveforms
        self.real_magnitude_spectrograms = real_magnitude_spectrograms
        self.fake_magnitude_spectrograms = fake_magnitude_spectrograms
        self.real_instantaneous_frequencies = real_instantaneous_frequencies
        self.fake_instantaneous_frequencies = fake_instantaneous_frequencies
        self.real_images = real_images
        self.fake_images = fake_images
        self.real_labels = labels
        self.fake_labels = labels
        self.real_features = real_features
        self.fake_features = fake_features
        self.real_logits = real_logits
        self.fake_logits = fake_logits
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_train_op = generator_train_op
        self.discriminator_train_op = discriminator_train_op

    def train(self, model_dir, config, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    ),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.audio(
                            name=name,
                            tensor=tensor,
                            sample_rate=16000,
                            max_outputs=4
                        ) for name, tensor in dict(
                            real_waveforms=self.real_waveforms,
                            fake_waveforms=self.fake_waveforms
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.image(
                            name=name,
                            tensor=tensor,
                            max_outputs=4
                        ) for name, tensor in dict(
                            real_magnitude_spectrograms=self.real_magnitude_spectrograms[..., tf.newaxis],
                            fake_magnitude_spectrograms=self.fake_magnitude_spectrograms[..., tf.newaxis],
                            real_instantaneous_frequencies=self.real_instantaneous_frequencies[..., tf.newaxis],
                            fake_instantaneous_frequencies=self.fake_instantaneous_frequencies[..., tf.newaxis]
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(
                            name=name,
                            tensor=tensor
                        ) for name, tensor in dict(
                            generator_loss=self.generator_loss,
                            discriminator_loss=self.discriminator_loss
                        ).items()
                    ]),
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        generator_loss=self.generator_loss,
                        discriminator_loss=self.discriminator_loss
                    ),
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                try:
                    session.run(self.discriminator_train_op)
                    session.run(self.generator_train_op)
                except tf.errors.OutOfRangeError:
                    break

    def evaluate(self, model_dir, config, classifier, input_name, output_names):

        real_features, real_logits = tf.import_graph_def(
            graph_def=classifier,
            input_map={input_name: self.real_images},
            return_elements=output_names
        )

        fake_features, fake_logits = tf.import_graph_def(
            graph_def=classifier,
            input_map={input_name: self.fake_images},
            return_elements=output_names
        )

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            def generator():
                while not session.should_stop():
                    try:
                        yield session.run([real_features, fake_features])
                    except tf.errors.OutOfRangeError:
                        break

            frechet_inception_distance = metrics.frechet_inception_distance(*map(np.concatenate, zip(*generator())))
            return dict(frechet_inception_distance=frechet_inception_distance)

    def generate(self, model_dir, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            while not session.should_stop():
                try:
                    yield session.run(self.fake_waveforms)
                except tf.errors.OutOfRangeError:
                    break


class PitchClassifier(object):

    def __init__(self, network, input_fn, spectral_params, hyper_params):

        waveforms, labels = input_fn()

        magnitude_spectrograms, instantaneous_frequencies = spectral_ops.convert_to_spectrogram(waveforms, **spectral_params)
        images = tf.stack([magnitude_spectrograms, instantaneous_frequencies], axis=1)

        features, logits = network(images)

        loss = tf.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=labels
        )
        loss += tf.add_n([
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
            if "normalization" not in variable.name
        ]) * hyper_params.weight_decay

        accuracy, update_op = tf.metrics.accuracy(
            predictions=tf.argmax(logits, axis=-1),
            labels=tf.argmax(labels, axis=-1)
        )

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=(
                hyper_params.learning_rate(tf.train.get_or_create_global_step())
                if callable(hyper_params.learning_rate) else hyper_params.learning_rate
            ),
            momentum=hyper_params.momentum,
            use_nesterov=hyper_params.use_nesterov
        )
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_or_create_global_step()
            )

        self.waveforms = waveforms
        self.magnitude_spectrograms = magnitude_spectrograms
        self.instantaneous_frequencies = instantaneous_frequencies
        self.loss = loss
        self.accuracy = accuracy
        self.train_op = train_op
        self.update_op = update_op

        images = tf.placeholder(tf.float32, shape=[None, *images.shape[1:]], name="images")
        features, logits = network(images)
        features = tf.identity(features, name="features")
        logits = tf.identity(logits, name="logits")

    def train(self, model_dir, config, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    ),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.audio(
                            name=name,
                            tensor=tensor,
                            sample_rate=16000,
                            max_outputs=4
                        ) for name, tensor in dict(
                            waveforms=self.waveforms
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.image(
                            name=name,
                            tensor=tensor,
                            max_outputs=4
                        ) for name, tensor in dict(
                            magnitude_spectrograms=self.magnitude_spectrograms[..., tf.newaxis],
                            instantaneous_frequencies=self.instantaneous_frequencies[..., tf.newaxis]
                        ).items()
                    ]),
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(
                            name=name,
                            tensor=tensor
                        ) for name, tensor in dict(
                            loss=self.loss,
                            accuracy=self.accuracy
                        ).items()
                    ]),
                ),
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        global_step=tf.train.get_global_step(),
                        loss=self.loss,
                        accuracy=self.accuracy
                    ),
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                try:
                    session.run([self.train_op, self.update_op])
                except tf.errors.OutOfRangeError:
                    break

    def evaluate(self, model_dir, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config
        ) as session:

            while not session.should_stop():
                try:
                    accuracy = session.run(self.update_op)
                except tf.errors.OutOfRangeError:
                    break

            return dict(accuracy=accuracy)
