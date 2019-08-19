from keras.models import load_model
from keras.optimizers import Adam
# from data_loader import DataLoader
import loadh5
import datetime
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np
import cv2

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# ktf.set_session(session)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('sr_name', None, 'name of the model')
flags.DEFINE_integer('epochs', 100, 'training epochs')
flags.DEFINE_string('dataset_name', 'BSDS500', 'dataset for training')


optimizer_sr = Adam(0.0001)
sr_model = load_model('saved_model/{}.h5'.format(FLAGS.sr_name))
com_model = load_model('saved_model/{}.h5'.format(FLAGS.sr_name))

dataset_name = FLAGS.dataset_name

# training
# start_time = datetime.datetime.now()
epoch = FLAGS.epochs


def np_PSNR(y_true, y_pred):
    diff = y_true - y_pred
    rmse = np.sqrt(np.mean(diff ** 2))
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / rmse)


def extract_Ydata(data):
    temp = []
    for num in range(data.shape[0]):
        img = cv2.cvtColor(data[num], cv2.COLOR_RGB2YCrCb)
        img_Y, _, _ = cv2.split(img)
        temp.append(img_Y)
    temp = np.array(temp)
    temp = temp.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return temp


hr, lr = loadh5.load_x_y_from_h5(dataset_name, 1, 128, 128, 3)
hr = extract_Ydata(hr) / 255.0
lr = extract_Ydata(lr) / 255.0


for epo in range(epoch):

    # Train the sr
    sr_model.fit(lr, hr, batch_size=16, epochs=1)
    
    # elapsed_time = datetime.datetime.now() - start_time

    # calculate psnr
    index = np.random.choice(hr.shape[0], size=1)
    test_hr = hr[index] 
    test_lr = lr[index]
    fake_hr = sr_model.predict(test_lr)
    com_hr = com_model.predict(test_lr)
    psnr = np_PSNR(test_hr, fake_hr)
    com_psnr = np_PSNR(test_hr, com_hr)
    print("PSNR: %f, com_PSNR: %f, develop: %f(%f)" % (psnr, com_psnr, psnr - com_psnr, (psnr - com_psnr) / com_psnr))


sr_model.save('saved_model/{}.h5'.format(FLAGS.sr_name))


