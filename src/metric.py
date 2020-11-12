import keras
import utils
import tensorflow as tf


def custom_acc(y_true, y_pred):
    y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
    pred_acc = keras.metrics.categorical_accuracy(y_true, y_pred_mean)
    return pred_acc


def custom_uc(y_true, y_pred):
    y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
    y_pred = tf.math.argmax(y_pred_mean, axis=1)  # extract class with highest score for each sample
    range_tensor = tf.convert_to_tensor(range(utils.batch_size), dtype=tf.int64)
    pred_uc = tf.gather_nd(y_pred_uc, tf.transpose([range_tensor, y_pred]))
    return pred_uc
