import keras
import tensorflow as tf
from network import *
import utils
import loss
import metric

learning_rate = 10e-5
decay_steps = 1000
decay_rate = 0.8


def get_vanilla_model():
    (inp, out) = get_layers(False)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                              decay_steps=decay_steps,
                                                              decay_rate=decay_rate)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])
    return model


def get_bcnn_model():
    (inp, out) = get_layers(True)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                              decay_steps=decay_steps,
                                                              decay_rate=decay_rate)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])
    return model


def get_custom_model(loss_fn):
    (inp, out) = get_layers(True)
    model = CustomModel(inputs=inp, outputs=out)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                              decay_steps=decay_steps,
                                                              decay_rate=decay_rate)

    model.compile(loss=loss_fn,
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=[metric.custom_acc, metric.custom_uc])
    return model


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            # HERE IS THE CUSTOMIZATION of our model:
            # we perform the forward pass T times, each time with a
            # different network configuration, due to dropout.
            # At the end we compute a single uncertainty measure, that
            # we use to compute the loss of the forward pass

            # we store the T predictions in a matrix with shape (T,B,K)
            pred_matrix = tf.zeros([utils.t, utils.batch_size, data_setup.num_classes], tf.float64)

            for i in range(utils.t):
                y_pred = self(x, training=True)  # Forward pass
                pred_matrix = tf.concat(axis=0, values=[pred_matrix[:i], [y_pred], pred_matrix[i + 1:]])

            # Compute the loss value according to my_loss_fct
            loss = self.compiled_loss(y, pred_matrix)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, pred_matrix)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # we store the T predictions in a matrix with shape (T,B,K)
        pred_matrix = tf.zeros([utils.t, utils.batch_size, data_setup.num_classes], tf.float64)

        for i in range(utils.t):
            y_pred = self(x, training=False)  # Forward pass
            pred_matrix = tf.concat(axis=0, values=[pred_matrix[:i], [y_pred], pred_matrix[i + 1:]])

        # Compute the loss value according to my_loss_fct
        self.compiled_loss(y, pred_matrix)

        # Update the metrics.
        self.compiled_metrics.update_state(y, pred_matrix)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

