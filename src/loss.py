# we define our custom loss function here
# y_pred is the matrix of all T predictions for each sample
def my_loss_fct(y_true, y_pred):
    loss_matrix = tf.zeros([num_predictions, batch_size], tf.float64)
    for i in range(num_predictions):
        pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred[i]) / 2 + 1;
        loss_matrix = tf.concat(axis=0, values=[loss_matrix[:i], [pred_loss], loss_matrix[i + 1:]])

    # TODO: now the final loss is just the average over the T losses (one loss per predicion),
    # we have to include the uncertainty quantification into the loss
    return tf.reduce_mean(loss_matrix, axis=0)