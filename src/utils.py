import numpy as np
import keras
from sklearn.metrics import accuracy_score


# Hyperparameters
batch_size = 32
epochs = 20


def train_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=30, verbose=1,
              validation_data=(x_val, y_val))


# This function computes the accuracy and the uncertainty of a model wrt a set of samples
def evaluate_model(model, x, y, num_classes, num_predictions):
    # predict real samples to compute 1. and 2.
    n_samples = x.shape[0]
    p_hat = np.zeros((num_predictions, n_samples, num_classes))
    for i in range(num_predictions):
        p_hat[i] = model.predict(x)

    p_hat = np.transpose(p_hat, [1, 0, 2])  # change shape to (n_samples, n_predictions, n_classes)
    p_hat_avg = np.mean(p_hat, axis=1)  # avg score for each class, with shape (n_samples, n_classes)
    y_pred = np.argmax(p_hat_avg,
                       axis=1)  # extract class with highest score for each sample, wich shape (n_samples,)

    y_pred_one_hot = keras.utils.to_categorical(y_pred, num_classes)
    acc = accuracy_score(y, y_pred_one_hot)
    # Compute uncertainty
    epistemic = np.mean(p_hat ** 2, axis=1) - np.mean(p_hat, axis=1) ** 2
    aleatoric = np.mean(p_hat * (1 - p_hat), axis=1)
    uncertainty = epistemic + aleatoric  # with shape (n_samples, n_classes)
    uc = uncertainty[
        range(n_samples), y_pred]  # for each samples we consider the uncertainty of the predicted class
    uc = np.mean(uc)  # compute a single scalar for all samples
    return acc, uc
