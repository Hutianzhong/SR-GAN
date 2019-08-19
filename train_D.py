from __future__ import print_function, division

import keras
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import datetime
import numpy as np
import keras.backend.tensorflow_backend as ktf
from D_model import D_bgan, D_srgan, D_acgan, D_bigan, D_ccgan, D_dcgan, D_gan, D_infogan, D_lsgan, D_wgan, D_sgan
import tensorflow as tf
import loadh5
import cv2
import random

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# ktf.set_session(session)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('gan_name', None, 'name of the GAN')
flags.DEFINE_string('sr_name', None, 'name of the model')
flags.DEFINE_integer('epochs', 100, 'training epochs')
flags.DEFINE_string('dataset_name', 'BSDS500', 'dataset for training')


# extract Y data from RGB
def extract_Ydata(data):
    temp = []
    for num in range(data.shape[0]):
        img = cv2.cvtColor(data[num], cv2.COLOR_RGB2YCrCb)
        img_Y, _, _ = cv2.split(img)
        temp.append(img_Y)
    temp = np.array(temp)
    temp = temp.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return temp


# Input shape
lr_shape = (128, 128, 1)
hr_shape = (128, 128, 1)

# Load img
hr, lr = loadh5.load_x_y_from_h5(FLAGS.dataset_name, 1, 128, 128, 3)
hr = extract_Ydata(hr) / 255.0
lr = extract_Ydata(lr) / 255.0

# Calculate output shape of D
if FLAGS.gan_name == 'srgan':
    patch_height = int(hr_shape.shape[0] / 2 ** 4)
    patch_width = int(hr_shape.shape[1] / 2 ** 4)
    disc_patch = (patch_height, patch_width, 1)
else:
    disc_patch = (1,)

# Number of filters in the first layer of D
df = 64


def return_model():
    return {
        'srgan': D_srgan.build_D(df),
        'bgan': D_bgan.build_D(hr_shape),
        'acgan': D_acgan.build_D(hr_shape),
        'bigan': D_bigan.build_D(hr_shape),
        'ccgan': D_ccgan.build_D(hr_shape),
        'dcgan': D_dcgan.build_D(hr_shape),
        'gan': D_gan.build_D(hr_shape),
        'infogan': D_infogan.build_D(hr_shape),
        'sgan': D_sgan.build_D(hr_shape),
        'lsgan': D_lsgan.build_D(hr_shape),
        'wgan': D_wgan.build_D(hr_shape)
    }.get(FLAGS.gan_name, 'error')


def step_func(epoch):
    if epoch < 30:
        return 0.0001
    elif epoch < 70:
        return 0.00001
    else:
        return 0.000001


# Build the discriminator
# optimizer_D = Adam(0.00001)
discriminator = return_model()
lrate = keras.callbacks.LearningRateScheduler(step_func,verbose=1)
discriminator.compile(loss='binary_crossentropy', optimizer='Adam')

# Load the vdsr
sr_model = load_model('saved_model/{}.h5'.format(FLAGS.sr_name))

# start_time = datetime.datetime.now()eee
# epoch = FLAGS.epochs

fake_hr = sr_model.predict(lr)
    
valid = np.ones((hr.shape[0],) + disc_patch) - np.random.random_sample((hr.shape[0],) + disc_patch) * 0.2
fake = np.random.random_sample((hr.shape[0],) + disc_patch) * 0.2

# Train the discriminator
train_sample = np.concatenate([fake_hr, hr], axis=0)
train_tar = np.concatenate([fake, valid], axis=0)

discriminator.fit(train_sample, train_tar, batch_size=4, epochs=100, shuffle=True, callbacks=[lrate])

# elapsed_time = datetime.datetime.now() - start_time

discriminator.save("saved_model/D_{}_Net.h5".format(FLAGS.gan_name))