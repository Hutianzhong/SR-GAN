from keras.models import load_model
from data_loader import DataLoader
import numpy as np
import cv2
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('sr_name', None, 'name of the model')
flags.DEFINE_string('dataset_name', 'test', 'dataset for testing')

dataset_name = FLAGS.dataset_name

data_loader = DataLoader(dataset_name=dataset_name)

imgs_hr, imgs_lr, imgs_real = data_loader.load_data(batch_size=1, is_inference=True)

vdsr = load_model('saved_model/{}.h5'.format(FLAGS.sr_name))
vdsr1 = load_model('saved_model/{}.h5'.format(FLAGS.sr_name))

v_hr = vdsr.predict(imgs_lr).reshape(imgs_real.shape[0], imgs_real.shape[1], 1)
v_hr1 = vdsr1.predict(imgs_lr).reshape(imgs_real.shape[0], imgs_real.shape[1], 1)
imgs_lr = imgs_lr.reshape(imgs_real.shape[0], imgs_real.shape[1], 1)
imgs_hr = imgs_hr.reshape(imgs_real.shape[0], imgs_real.shape[1], 1)

v_ans = np.zeros((imgs_real.shape[0], imgs_real.shape[1], 3))
v1_ans = np.zeros((imgs_real.shape[0], imgs_real.shape[1], 3))
lr = np.zeros((imgs_real.shape[0], imgs_real.shape[1], 3))
hr = np.zeros((imgs_real.shape[0], imgs_real.shape[1], 3))

v_ans[:, :, :1] = v_hr * 255
v_ans[:, :, 1:3] = imgs_real[:, :, 1:3]
v1_ans[:, :, :1] = v_hr1 * 255
v1_ans[:, :, 1:3] = imgs_real[:, :, 1:3]
lr[:, :, :1] = imgs_lr * 255
lr[:, :, 1:3] = imgs_real[:, :, 1:3]
hr[:, :, :1] = imgs_hr * 255
hr[:, :, 1:3] = imgs_real[:, :, 1:3]

v_ans = v_ans.clip(0, 255)
v1_ans = v1_ans.clip(0, 255)

lr = lr.astype(np.uint8)
v_ans = v_ans.astype(np.uint8)
v1_ans = v1_ans.astype(np.uint8)
hr = hr.astype(np.uint8)

v_ans = cv2.cvtColor(v_ans, cv2.COLOR_YCR_CB2RGB)
v1_ans = cv2.cvtColor(v1_ans, cv2.COLOR_YCR_CB2RGB)
lr = cv2.cvtColor(lr, cv2.COLOR_YCR_CB2RGB)
hr = cv2.cvtColor(hr, cv2.COLOR_YCR_CB2RGB)


def np_PSNR(y_true, y_pred):
    diff = y_true - y_pred
    rmse = np.sqrt(np.mean(diff ** 2))
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / rmse)


psnr = np_PSNR(hr/255, v1_ans/255)
com_psnr = np_PSNR(hr/255, v_ans/255)
print("PSNR: %f, com_PSNR: %f, develop: %f(%f)" % (psnr, com_psnr, psnr - com_psnr, (psnr - com_psnr) / com_psnr))

out = np.hstack([lr, v_ans, v1_ans, hr])

cv2.imwrite('images/%s/%s_out.jpg' % (dataset_name, FLAGS.sr_name), out)