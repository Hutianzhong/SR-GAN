# -*- coding: utf-8 -*-

from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense


def build_D(fliters=64, hr_shape=(None, None, 1)):

    def d_block(layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    # Input img
    d0 = Input(shape=hr_shape)

    d1 = d_block(d0, fliters, bn=False)
    d2 = d_block(d1, fliters, strides=2)
    d3 = d_block(d2, fliters * 2)
    d4 = d_block(d3, fliters * 2, strides=2)
    d5 = d_block(d4, fliters * 4)
    d6 = d_block(d5, fliters * 4, strides=2)
    d7 = d_block(d6, fliters * 8)
    d8 = d_block(d7, fliters * 8, strides=2)

    d9 = Dense(fliters * 16)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(d0, validity)