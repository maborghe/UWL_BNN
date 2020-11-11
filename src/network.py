from keras.models import Sequential, Model, Input
from keras.layers import Dense, InputLayer, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, Add, BatchNormalization

# Network architecture
def get_dropout(input_tensor, p=0.5, mc=False):
    if mc:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)

def getLayers(num_classes, mc=False):
    inp = Input(input_shape)
    # x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')(inp)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = get_dropout(x, p=0.25, mc=mc)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = get_dropout(x, p=0.5, mc=mc)
    out = Dense(num_classes, activation='softmax')(x)
    return (inp, out)