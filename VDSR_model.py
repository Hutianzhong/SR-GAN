#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers.convolutional import Conv2D
from keras.layers import Input, Activation, add
from keras.models import Model

from keras import backend as K
K.set_image_data_format('channels_last')


def VDSR():
    input_img = Input(shape=(None, None, 1))
    model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    model = Activation('relu')(model)
    for i in range(18):
        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = Activation('relu')(model)
    model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
    res_img = model
    output_img = add([res_img, input_img])
    model = Model(input_img, output_img)
    return model
