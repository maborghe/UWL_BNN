from keras.models import Sequential, Model, Input
from keras.layers import Dense, InputLayer, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, Add, BatchNormalization
import data_setup


# Network architecture
def get_dropout(input_tensor, p, mc):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


# def get_layers(mc):
#     inp = Input(data_setup.input_shape)
#     x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')(inp)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = BatchNormalization()(x)
#     x = get_dropout(x, p=0.25, mc=mc)
#     x = Flatten()(x)
#     x = Dense(128, activation='relu')(x)
#     x = get_dropout(x, p=0.5, mc=mc)
#     out = Dense(data_setup.num_classes, activation='softmax')(x)
#     return inp, out

def get_layers(mc):
    inp = Input(data_setup.input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')(inp)
    x = get_dropout(x, p=0.25, mc=mc)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = BatchNormalization()(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = get_dropout(x, p=0.5, mc=mc)
    out = Dense(data_setup.num_classes, activation='softmax')(x)
    return inp, out
