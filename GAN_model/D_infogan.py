from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model


def build_D(hr_shape=(128, 128, 1)):

    img = Input(shape=hr_shape)

    # Shared layers between discriminator and recognition network
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=hr_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())

    img_embedding = model(img)

    # Discriminator
    validity = Dense(1, activation='sigmoid')(img_embedding)

    # Return discriminator
    return Model(img, validity)