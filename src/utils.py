import numpy as np
import keras
import tensorflow as tf
import data_setup
from sklearn.metrics import accuracy_score
import keras.backend as K

# Hyperparameters
batch_size = 32
t = 10  # a.k.a. num predictions


def train_model(model, x_train, y_train, x_val, y_val, epochs):
    # We cut the samples so that it is a multiple of the batch size
    # TODO: It's not necessary to do that!
    rem_train = x_train.shape[0] % batch_size
    if rem_train != 0:
        x_train = x_train[:-rem_train]
        y_train = y_train[:-rem_train]
    rem_val = x_val.shape[0] % batch_size
    if rem_val != 0:
        x_val = x_val[:-rem_val]
        y_val = y_val[:-rem_val]

    with tf.device("gpu:0"):
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                  validation_data=(x_val, y_val))


# This function computes the accuracy and the uncertainty of a model wrt a set of samples
def evaluate_baseline(model, x, y, multiple_pred):
    num_predictions = t if multiple_pred else 1

    # predict real samples to compute 1. and 2.
    n_samples = x.shape[0]
    p_hat = np.zeros((num_predictions, n_samples, data_setup.num_classes))
    for i in range(num_predictions):
        p_hat[i] = model.predict(x)

    y_pred_mean, y_pred_uc = compute_pred_distribution(p_hat)

    y_pred = np.argmax(y_pred_mean, axis=1)  # extract class with highest score for each sample
    y_pred_one_hot = keras.utils.to_categorical(y_pred, data_setup.num_classes)
    acc = accuracy_score(y, y_pred_one_hot)

    # for each samples we consider the uncertainty of the predicted class
    uc = tf.gather_nd(y_pred_uc, tf.transpose([range(n_samples), y_pred]))
    uc = np.mean(uc)  # compute a single scalar for all samples
    return acc, uc


def evaluate_custom(model, x, y):
    # We cut the samples so that it is a multiple of the batch size
    rem = x.shape[0] % batch_size
    if rem != 0:
        x = x[:-rem]
        y = y[:-rem]
    return model.evaluate(x, y, batch_size=batch_size)


# y_true has shape (n_samples, n_classes)
# y_pred has shape (num_predictions, n_samples, n_classes)
# y_pred is the matrix of all T predictions for each sample
def compute_pred_distribution(y_pred):
    y_pred = tf.transpose(y_pred, [1, 0, 2])  # change shape to (batch_size, num_predictions, num_classes)
    # 1. Compute mean
    y_pred_mean = tf.math.reduce_mean(y_pred, axis=1)  # avg score for each class, with shape (n_samples, num_classes)

    # 2. Compute uncertainty
    epistemic = tf.reduce_mean(y_pred ** 2, axis=1) - tf.reduce_mean(y_pred, axis=1) ** 2
    aleatoric = tf.reduce_mean(y_pred * (1 - y_pred), axis=1)
    y_pred_uc = epistemic + aleatoric  # with shape (n_samples, n_classes)
    return y_pred_mean, y_pred_uc  # each with shape (n_samples, n_classes)
