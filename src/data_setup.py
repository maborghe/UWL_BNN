import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import keras
import random
import math

num_channels = 3
num_classes = 3
img_rows, img_cols = 32, 32
input_shape = None
mean_img = None


def get_train_data():
    global mean_img
    global input_shape

    x_train = np.load('../dataset/X_train.npy')
    y_train = np.load('../dataset/Y_train.npy')
    x_train_flip = np.load('../dataset/X_train_flip.npy')

    real_classes_train = np.load('../dataset/real_classes_train.npy')
    real_classes_train_ids = [i for i, label in enumerate(real_classes_train) if label in ['AH', 'AD', 'H']]
    print('Train samples: ' + str(x_train.shape[0]))
    print('Train samples: ' + str(x_train.shape[0]))
    # plt.imshow(x_train[0])

    x_train = np.concatenate((x_train, x_train_flip))
    y_train = np.concatenate((y_train, y_train))
    # Preprocess images
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], num_channels, img_rows, img_cols)
        input_shape = (num_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
        input_shape = (img_rows, img_cols, num_channels)
    x_train = x_train.astype('float32')

    # As in AlexNet, with subtract the mean image to have zero mean in the training set
    mean_img = np.mean(x_train, axis=0, keepdims=True)
    x_train = x_train - mean_img

    # Preprocess labels
    y_train = keras.utils.to_categorical(y_train, num_classes)

    # Build "real" sets, i.e. sets consisting only of correctly labeled items
    # Training set
    x_train_real = x_train[real_classes_train_ids]
    x_train_flip_real = x_train_flip[real_classes_train_ids]
    y_train_real = y_train[real_classes_train_ids]
    x_train_real = np.concatenate((x_train_real, x_train_flip_real))
    y_train_real = np.concatenate((y_train_real, y_train_real))

    print('x_train shape:', x_train.shape)
    print('Training samples: ', x_train.shape[0])
    print('Real training samples: ', x_train_real.shape[0])

    return x_train, y_train, x_train_real, y_train_real


def get_dev_test_data():
    if mean_img is None:
        raise Exception("Mean image not available. Please call get_train_data first")

    x_test_orig = np.load('../dataset/X_test.npy') # orig because it will be split to obtain the validation set
    y_test_orig = np.load('../dataset/Y_test.npy')
    real_classes_test = np.load('../dataset/real_classes_test.npy')
    real_classes_test_ids = [i for i, label in enumerate(real_classes_test) if label in ['AH', 'AD', 'H']]
    print('Test samples: ' + str(x_test_orig.shape[0]))
    print('Real test samples: ' + str(len(real_classes_test_ids)))

    # Preprocess images
    if K.image_data_format() == 'channels_first':
        x_test_orig = x_test_orig.reshape(x_test_orig.shape[0], num_channels, img_rows, img_cols)
    else:
        x_test_orig = x_test_orig.reshape(x_test_orig.shape[0], img_rows, img_cols, num_channels)

    x_test_orig = x_test_orig.astype('float32')
    x_test_orig = x_test_orig - mean_img

    y_test_orig = keras.utils.to_categorical(y_test_orig, num_classes)

    # Build validation sets from the test distribution
    # Split original test set into two equal sets, one for validation and one for testing
    n_test = x_test_orig.shape[0]
    val_ids = random.sample(range(n_test), math.floor(0.5 * n_test))
    test_ids = [n for n in range(n_test) if n not in val_ids]
    x_val = x_test_orig[val_ids]
    x_test = x_test_orig[test_ids]
    y_val = y_test_orig[val_ids]
    y_test = y_test_orig[test_ids]

    # Val/test set
    val_real_ids = [id for id in val_ids if id in real_classes_test_ids]
    test_real_ids = [id for id in test_ids if id in real_classes_test_ids]
    val_fake_ids = [id for id in val_ids if id not in real_classes_test_ids]
    test_fake_ids = [id for id in test_ids if id not in real_classes_test_ids]
    x_val_real = x_test_orig[val_real_ids]
    y_val_real = y_test_orig[val_real_ids]
    x_test_real = x_test_orig[test_real_ids]
    y_test_real = y_test_orig[test_real_ids]
    x_val_fake = x_test_orig[val_fake_ids]
    y_val_fake = y_test_orig[val_fake_ids]
    x_test_fake = x_test_orig[test_fake_ids]
    y_test_fake = y_test_orig[test_fake_ids]

    print('Validation samples: ', x_val.shape[0])
    print('Test samples: ', x_test.shape[0])
    print('Real validation samples: ', x_val_real.shape[0])
    print('Real test samples: ', x_test_real.shape[0])

    return x_val_real, y_val_real, x_val_fake, y_val_fake,\
           x_test_real, y_test_real, x_test_fake, y_test_fake


def augment_data():
    # X = np.concatenate((X_train, X_test))
    # n_tot_samples = X.shape[0]
    # for i in range(n_tot_samples):
    #  sample = X[i]
    #  res = np.flip(sample, axis=1) # mirror image
    #  X = np.append(X, [res], axis=0)

    # X_flip = X[n_tot_samples:]
    # n_train_samples = X_train.shape[0]
    # X_train_flip = X_flip[:n_train_samples]
    # Y_train_flip = Y_train
    # X_test_flip = X_flip[n_train_samples:]
    # Y_test_flip = Y_test
    return 0
