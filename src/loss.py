import utils
import keras
import tensorflow as tf


# we define our custom loss function here
# y_pred is the matrix of all T predictions for each sample
def my_loss_fct(y_true, y_pred):
    y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)

    # TODO: now we have to arrays: y_pred_mean contains the class scores for each sample
    # while y_pred_uc contains the uncertainty of that score.
    # We have to take out a single scalar as the loss (e.g. we could compute the cross-entropy for
    # each sample and then return a mean weighted by the uncertainty)
    pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred_mean)
    return pred_loss


def my_acc(y_true, y_pred):
    y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
    pred_acc = keras.metrics.categorical_accuracy(y_true, y_pred_mean)
    return pred_acc
