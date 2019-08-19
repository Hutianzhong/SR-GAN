import scipy
from glob import glob
import numpy as np
import cv2


class DataLoader():
    def __init__(self, dataset_name, img_res=(480, 320)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_inference=False):

        path = glob('./datasets/%s/*.jpg' % (self.dataset_name))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        
        for img_path in batch_images:
            img = self.imread(img_path)

            if img.shape[0] < img.shape[1]:
                img = img.transpose(1, 0, 2)
            
            # the resized array of image
            # img = scipy.misc.imresize(img, (480, 320, 3), interp='bicubic')
            img = img[:480, :320]
            img_y, _, _ = cv2.split(img)
            
            if is_inference:
                img_hr = img_y
                img_lr = cv2.resize(img_hr, (img_hr.shape[1]//2, img_hr.shape[0]//2), interpolation=cv2.INTER_CUBIC)
                img_lr = cv2.resize(img_lr, (img_hr.shape[1], img_hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                img_hr = scipy.misc.imresize(img_y, 1 / 4, interp='bicubic') # (120, 80)
                img_lr = scipy.misc.imresize(img_hr, 1 / 2, interp='bicubic') # (60, 40)
                img_lr = scipy.misc.imresize(img_lr, img_hr.shape, interp='bicubic') # (120, 80)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 255.0
        imgs_lr = np.array(imgs_lr) / 255.0
        imgs_hr.resize([batch_size, img_hr.shape[0], img_hr.shape[1], 1])
        imgs_lr.resize([batch_size, img_lr.shape[0], img_lr.shape[1], 1])

        return imgs_hr, imgs_lr, img

    def imread(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)