# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2021/4/1 19:48
# @Site        : xxx#2L4
# @File         : load_data
# @Software : PyCharm
# @Author   : Dave Liu
# @Email    :
# @Version  : V1.1.0
------------------------------------------------- 
"""
import scipy.io as sio
import h5py


def loading_data(path):
    # path = '/home/user045/linqiubin/Datasets/DCMH/mirflickr-25k/'
    # load original image : (20015, 3, 244, 244)
    IALL = h5py.File(path + 'mirflickr25k-iall.mat', 'r')
    images = IALL['IAll'][:]

    # load text feature : (20015, 1386)
    YALL = sio.loadmat(path + 'mirflickr25k-yall.mat')
    tags = YALL['YAll'][:]

    # load label : (20015, 24)
    LALL = sio.loadmat(path + 'mirflickr25k-lall.mat')
    labels = LALL['LAll'][:]

    IALL.close()

    print('images: %s' % str(images.shape))
    print('text: %s' % str(tags.shape))
    print('labels: %s' % str(labels.shape))

    return images, tags, labels


if __name__ == '__main__':
    images, tags, labels = loading_data()