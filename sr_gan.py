from __future__ import print_function, division
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import datetime
from data_loader import DataLoader
import numpy as np
import os
import keras.backend.tensorflow_backend as ktf
from D_model import D_bgan, D_srgan, D_acgan, D_bigan, D_ccgan, D_dcgan, D_gan, D_infogan, D_lsgan, D_wgan, D_sgan
import tensorflow as tf
import loadh5
import cv2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('gan_name', None, 'name of the GAN')
flags.DEFINE_string('sr_name', None, 'name of the model')
flags.DEFINE_integer('epochs', 10, 'training epochs')
flags.DEFINE_string('dataset_name', 'BSDS500', 'dataset for training')
flags.DEFINE_integer('batch_size', 1, 'batch size of training data')

class GAN():
    def __init__(self):
        
        # Configure data loader
        self.dataset_name = FLAGS.dataset_name
        
        # Input shape
        self.lr_shape = (128, 128, 1)
        self.hr_shape = (128, 128, 1)

        # Calculate output shape of D (PatchGAN)

        if FLAGS.gan_name == 'srgan':
            patch_height = int(self.hr_shape.shape[0] / 2 ** 4)
            patch_width = int(self.hr_shape.shape[1] / 2 ** 4)
            self.disc_patch = (patch_height, patch_width, 1)
        else:
            self.disc_patch = (1,)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        
        optimizer_G = Adam(0.0001)  # lr and beta
        optimizer_D = Adam(0.00001)

        # Build the discriminator
        self.discriminator = load_model("saved_model/D_{}_Net.h5".format(FLAGS.gan_name))
        
        # compare_model
        self.com_model = load_model('saved_model/{}.h5'.format(FLAGS.sr_name))

        # Build the generator
        self.sr_model = load_model('saved_model/{}.h5'.format(FLAGS.sr_name))

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)
        fake_hr = Input(shape=self.hr_shape)
        
        # G_Net
        fake_hr = self.sr_model(img_lr)
        fake_validity = self.discriminator(fake_hr)
        self.G_net = Model(img_lr, [fake_hr, fake_validity])
        self.G_net.compile(loss=['mse', 'binary_crossentropy'], optimizer=optimizer_G)
    
    def set_trainability(self, model, trainable=False):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    def return_model(self):
        return {
            'srgan': D_srgan.build_D(self.df),
            'bgan': D_bgan.build_D(self.hr_shape),
            'acgan': D_acgan.build_D(self.hr_shape),
            'bigan': D_bigan.build_D(self.hr_shape),
            'ccgan': D_ccgan.build_D(self.hr_shape),
            'dcgan': D_dcgan.build_D(self.hr_shape),
            'gan': D_gan.build_D(self.hr_shape),
            'infogan': D_infogan.build_D(self.hr_shape),
            'sgan': D_sgan.build_D(self.hr_shape),
            'lsgan': D_lsgan.build_D(self.hr_shape),
            'wgan': D_wgan.build_D(self.hr_shape)
        }.get(FLAGS.gan_name, 'error')

    def extract_Ydata(self, data):
        temp = []
        for num in range(data.shape[0]):
            img = cv2.cvtColor(data[num], cv2.COLOR_RGB2YCrCb)
            img_Y, _, _ = cv2.split(img)
            temp.append(img_Y)
        temp = np.array(temp)
        temp = temp.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
        return temp

    def train(self, epochs, batch_size=1, sample_interval=50):
        
        start_time = datetime.datetime.now()

        # Load img
        hr, lr = loadh5.load_x_y_from_h5(FLAGS.dataset_name, 1, 128, 128, 3)
        hr = self.extract_Ydata(hr) / 255.0
        lr = self.extract_Ydata(lr) / 255.0

        for epoch in range(epochs):
        
            for num in range(0, hr.shape[0], batch_size):
               
                imgs_lr = lr[num:num+batch_size, :, :, :]
                imgs_hr = hr[num:num+batch_size, :, :, :]
                
                # ----------------------
                #  Train Discriminator
                # ----------------------
    
                self.set_trainability(self.discriminator, True)
                self.set_trainability(self.sr_model, False)
                                
                fake_hr = self.sr_model.predict(imgs_lr)
                valid = np.ones((batch_size,) + self.disc_patch) - np.random.random_sample((batch_size,) + self.disc_patch) * 0.2
                fake = np.random.random_sample((batch_size,) + self.disc_patch) * 0.2

                # Train the discriminator
                train_sample = np.concatenate([fake_hr, imgs_hr], axis=0)
                train_tar = np.concatenate([fake, valid], axis=0)
                
                d_loss = self.discriminator.train_on_batch(train_sample, train_tar)

                # ------------------
                #  Train Generator
                # ------------------

                self.set_trainability(self.discriminator, False)
                self.set_trainability(self.sr_model, True)

                # Train the generators
                g_loss = self.G_net.train_on_batch(imgs_lr, [imgs_hr, valid])

                elapsed_time = datetime.datetime.now() - start_time

                # If at save interval => calculate loss and psnr
                if num % sample_interval == 0:
                    print("Epoch: %d time: %s d_loss: %s g_loss: %s" % (epoch, elapsed_time, str(d_loss), str(g_loss)))
                    self.sample_test(imgs_lr, imgs_hr)

        self.sr_model.save("saved_model/{}_{}_{}_Net.h5".format(FLAGS.sr_name, FLAGS.gan_name, batch_size))
        self.discriminator.save("saved_model/D_{}_{}_Net.h5".format(FLAGS.gan_name, batch_size))

    def sample_test(self, imgs_lr, imgs_hr):

        fake_hr = self.sr_model.predict(imgs_lr)
        com_hr = self.com_model.predict(imgs_lr)

        # calculate psnr
        psnr = self.np_PSNR(imgs_hr, fake_hr)
        com_psnr = self.np_PSNR(imgs_hr, com_hr)
        print("PSNR: %f, com_PSNR: %f, develop: %f(%f)" % (psnr, com_psnr, psnr - com_psnr, (psnr - com_psnr) / com_psnr))

    def np_PSNR(self, y_true, y_pred):
        diff = y_true - y_pred
        rmse = np.sqrt(np.mean(diff ** 2))
        max_pixel = 1.0
        return 20 * np.log10(max_pixel / rmse)


def running(args):
    gan = GAN()
    gan.train(epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, sample_interval=100)


if __name__ == '__main__':
    tf.app.run(running)