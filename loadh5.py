import h5py
import numpy as np


def load_x_y_from_h5(filename, down=2, height=64, width=64, channel=3):

    f = h5py.File(filename, 'r')
    key_x = list(f.keys())[0]
    key_y = list(f.keys())[1]

    # Get the data
    data_x = f[key_x][:]
    data_y = f[key_y][:]
    hr = data_x.reshape(-1, height, width, channel)
    lr = data_y.reshape(-1, height//down, width//down, channel)
    del data_x
    del data_y

    f.close()

    return hr, lr