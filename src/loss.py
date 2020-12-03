import utils
import keras


# we define our custom loss function here
# y_pred is the matrix of all T predictions for each sample
def my_loss_fct(y_true, y_pred):    
  y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
  pred_class = K.argmax(y_pred_mean, axis=-1)
  pred_class = tf.reshape(pred_class, [-1, 1])
  class_uc = tf.gather_nd(y_pred_uc, pred_class, batch_dims=1)  
  
  class_uc = 1 - class_uc
  class_uc_inv = class_uc  

  pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred_mean)*class_uc_inv;  # we weight the loss by the computed weights
  return pred_loss  


def my_loss_fct2(y_true, y_pred):
  
  y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
  pred_class = K.argmax(y_pred_mean, axis=-1)
  pred_class = tf.reshape(pred_class, [-1, 1])
  class_uc = tf.gather_nd(y_pred_uc, pred_class, batch_dims=1)  
  
  class_uc = K.pow(class_uc, 2)
  class_uc = 1 - class_uc
  class_uc_inv = class_uc  

  pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred_mean)*class_uc_inv;  # we weight the loss by the computed weights
  return pred_loss

def my_loss_fct3(y_true, y_pred):
  
  y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
  pred_class = K.argmax(y_pred_mean, axis=-1)
  pred_class = tf.reshape(pred_class, [-1, 1])
  class_uc = tf.gather_nd(y_pred_uc, pred_class, batch_dims=1)  
  
  class_uc = K.pow(class_uc, 3)
  class_uc = 1 - class_uc
  class_uc_inv = class_uc  

  pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred_mean)*class_uc_inv;  # we weight the loss by the computed weights
  return pred_loss  


def my_loss_fct4(y_true, y_pred):
    
  y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
  pred_class = K.argmax(y_pred_mean, axis=-1)
  pred_class = tf.reshape(pred_class, [-1, 1])
  class_uc = tf.gather_nd(y_pred_uc, pred_class, batch_dims=1)  
  
  class_uc /= keras.backend.mean(class_uc) 
  class_uc = K.exp(class_uc)
  class_uc_inv = 1 / class_uc  # in this way uncertain samples have a lower weight
  class_uc_inv /= keras.backend.mean(class_uc_inv)  # we normalize the weights so that they average to 1

  pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred_mean)*class_uc_inv;  # we weight the loss by the computed weights  
  return pred_loss  

def my_loss_fct5(y_true, y_pred):

  y_pred_mean, y_pred_uc = utils.compute_pred_distribution(y_pred)
  pred_class = K.argmax(y_pred_mean, axis=-1)
  pred_class = tf.reshape(pred_class, [-1, 1])
  class_uc = tf.gather_nd(y_pred_uc, pred_class, batch_dims=1)  
  
  class_uc = K.exp(class_uc)
  class_uc_inv = 1 / class_uc  # in this way uncertain samples have a lower weight
  class_uc_inv /= keras.backend.mean(class_uc_inv)  # we normalize the weights so that they average to 1

  pred_loss = keras.losses.categorical_crossentropy(y_true, y_pred_mean)*class_uc_inv;  # we weight the loss by the computed weights  
  return pred_loss  
