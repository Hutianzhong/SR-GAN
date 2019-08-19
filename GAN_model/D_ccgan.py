from __future__ import print_function, division

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model


def build_D(hr_shape=(128, 128, 1)):

    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=hr_shape))
    model.add(LeakyReLU(alpha=0.8))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(InstanceNormalization())
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(InstanceNormalization())

    model.summary()

    img = Input(shape=hr_shape)
    features = model(img)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(features)

    return Model(img, validity)