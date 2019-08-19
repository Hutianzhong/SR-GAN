from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model


def build_D(hr_shape=(128, 128, 1)):
    model = Sequential()

    model.add(Flatten(input_shape=hr_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=hr_shape)
    validity = model(img)

    return Model(img, validity)