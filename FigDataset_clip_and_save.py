import h5py
import os
import cv2
import numpy as np

cliped_h, cliped_w = 128, 128
down = 1


def clip(img):
    hr_list = []
    lr_list = []

    h = img.shape[0]
    w = img.shape[1]

    h_num = h // cliped_h
    w_num = w // cliped_w

    for i in range(h_num):
        for j in range(w_num):
            img_hr = img[i*cliped_h:(i+1)*cliped_h, j*cliped_w:(j+1)*cliped_w, :]
            if down==1:
                img_lr = cv2.resize(img_hr, (cliped_w//2, cliped_h//2), interpolation=cv2.INTER_CUBIC)
                img_lr = cv2.resize(img_lr, (cliped_w, cliped_h), interpolation=cv2.INTER_CUBIC)
            else:
                img_lr = cv2.resize(img_hr, (cliped_w//down, cliped_h//down), interpolation=cv2.INTER_CUBIC)
            hr_list.append(img_hr)
            lr_list.append(img_lr)

    return hr_list, lr_list


def get_files(file_dir):

    hr = []
    lr = []
    
    img_list = os.listdir(file_dir)
    
    for i, file in enumerate(img_list):
        str = os.path.splitext(file)[1]
        if str == '.png' or str == '.jpg':
            img = cv2.imread(file_dir + '/' + file)
            hr_list, lr_list = clip(img)
            hr.extend(hr_list)
            lr.extend(lr_list)
            print("Saving {}/{}".format(i, len(img_list)))

    hr = np.array(hr)
    hr = hr.reshape(-1)
    lr = np.array(lr)
    lr = lr.reshape(-1)

    h = h5py.File('BSDS500_x{}_{}x{}.h5'.format(down, cliped_h, cliped_w), 'w')
    h.create_dataset('hr', data=hr)
    h.create_dataset('lr', data=lr)
    h.close()
    print('Finish!')

path = './BSDS500'
get_files(path)
