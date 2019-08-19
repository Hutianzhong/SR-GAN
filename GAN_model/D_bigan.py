from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model


def build_D(hr_shape=(128, 128, 1), latent_dim=100):

    def build_encoder():
        model = Sequential()

        model.add(Flatten(input_shape=hr_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(latent_dim))

        model.summary()

        img = Input(shape=hr_shape)
        z = model(img)

        return Model(img, z)

    img = Input(shape=hr_shape)
    encoder = build_encoder()
    z = encoder(img)
    d_in = concatenate([z, Flatten()(img)])

    model = Dense(1024)(d_in)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation="sigmoid")(model)

    return Model(img, validity)