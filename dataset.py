# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2021/4/1 18:40
# @Site        : xxx#2L4
# @File         : dataset
# @Software : PyCharm
# @Author   : Dave Liu
# @Email    :
# @Version  : V1.1.0
------------------------------------------------- 
"""
import torch
#from datasets.data_handler import data_enhance

import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, images, tags, labels, test=None):
        self.test = test
        np.random.seed(0)
        index_all = np.random.permutation(opt.query_size + opt.database_size)
        ind_Q = index_all[0: opt.query_size]
        ind_T = index_all[opt.query_size: opt.query_size + opt.training_size]
        ind_R = index_all[opt.query_size: opt.query_size + opt.database_size]
        if test is None:
            train_images = images[ind_T]
            train_tags = tags[ind_T]
            train_labels = labels[ind_T]
            if opt.data_enhance:
                self.images, self.tags, self.labels = data_enhance(train_images, train_tags, train_labels)
            else:
                self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[ind_Q]
            self.db_labels = labels[ind_R]
            if test == 'image.query':
                self.images = images[ind_Q]
            elif test == 'image.db':
                self.images = images[ind_R]
            elif test == 'text.query':
                self.tags = tags[ind_Q]
            elif test == 'text.db':
                self.tags = tags[ind_R]
        # if hasattr(self, 'images'):
        #     self.images = preprocess(self.images, mean=(0.336, 0.324, 0.293), std=(0.182, 0.182, 0.190))

    def __getitem__(self, index):
        if self.test is None:
            return (
                index,
                torch.from_numpy(self.images[index].astype('float32')),
                torch.from_numpy(self.tags[index].astype('float32')),
                torch.from_numpy(self.labels[index].astype('float32'))
            )
        elif self.test.startswith('image'):
            return torch.from_numpy(self.images[index].astype('float32'))
        elif self.test.startswith('text'):
            return torch.from_numpy(self.tags[index].astype('float32'))

    def __len__(self):
        if self.test is None:
            return len(self.images)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return torch.from_numpy(self.labels.astype('float32'))
        else:
            return (
                torch.from_numpy(self.query_labels.astype('float32')),
                torch.from_numpy(self.db_labels.astype('float32'))
            )