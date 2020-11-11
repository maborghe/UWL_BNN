import keras
import tensorflow as tf
from network import *
from loss import *

learning_rate = 10e-5
decay_steps = 1000
decay_rate = 0.8
# num_predictions = 10  # aka 'T'


def get_vanilla_model():
    (inp, out) = getLayers(False)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                              decay_steps=decay_steps,
                                                              decay_rate=decay_rate)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])
    return model


def get_bcnn_model():
    (inp, out) = getLayers(True)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                              decay_steps=decay_steps,
                                                              decay_rate=decay_rate)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                  metrics=['accuracy'])
    return model


# we define our custom model here
class CustomModel(keras.Model):
    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass

    def __init__(self, batch_size, num_classes, num_predictions):
        # TODO: call superclass init
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_predictions = num_predictions

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
            pred_matrix = tf.zeros([self.num_predictions, self.batch_size, self.num_classes], tf.float64)

            for i in range(self.num_predictions):
                y_pred = self(x, training=True)  # Forward pass
                pred_matrix = tf.concat(axis=0, values=[pred_matrix[:i], [y_pred], pred_matrix[i + 1:]])

            # Compute the loss value according to my_loss_fct
            loss = my_loss_fct(y, pred_matrix)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def get_custom_model():
    (inp, out) = getLayers(True)
    model = CustomModel(inputs=inp, outputs=out)

    model.compile(optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
