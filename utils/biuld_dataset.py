#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy import signal

# 构建数据集
past_history = 120  # 历史时间序列长度
future_target = 0  # 预测的时间点
STEP = 1  # 滑窗移动步长

# we add the noise
"""
    'gauss'     Gaussian-distributed additive noise.
    'speckle'   out = image + n*image,where
                n is uniform noise with specified mean & variance.       
"""
def add_noise(dataset, noise_type="gaussian"):  # input includes the type of the noise to be added and the input image
    # dataset = dataset.astype(dataset.float32)

    if noise_type == "gaussian":
        noise = np.random.normal(-0.01, 0.1, dataset.shape)  # input includes : mean, deviation, shape of the image and the function picks up a normal distribuition.
        dataset = dataset + noise  # adding the noise
        return dataset
    if noise_type == "speckle":
        noise = np.random.randn(dataset.shape)
        img = dataset + dataset * noise
        return img

def dataset_noise(x_train, x_test):
    # Now dividing the dataset into two parts and adding gaussian to one and speckle to another.
    noises = ["gaussian", "speckle"]

    noise_id = 0  # id represnts which noise is being added, its 0 = gaussian and 1 = speckle
    traindata = np.zeros(x_train.shape)  # revised training data
    for idx in tqdm(range(len(x_train))):  # for the first half we are using gaussian noise & for the second half speckle noise
        traindata[idx] = add_noise(x_train[idx], noise_type=noises[noise_id])
    print("\n{} noise addition completed to images".format(noises[noise_id]))

    noise_id = 0
    testdata = np.zeros(x_test.shape)
    for idx in tqdm(range(len(x_test))):  # Doing the same for the test set.
        testdata[idx] = add_noise(x_test[idx], noise_type=noises[noise_id])
    print("\n{} noise addition completed to images".format(noises[noise_id]))

    return traindata, testdata

# data prepare
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False, stft_flag=True):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        if stft_flag:
            slice_ = dataset[indices]
            f, t, stftvib = signal.stft(slice_[:,0], 1, nperseg=60, noverlap=59)
            stftvib = np.abs(stftvib).T
            stftvib = stftvib[:len(slice_), :]
            slice_stft = np.concatenate([slice_, stftvib], axis=1)
            data.append(slice_stft)
        else:
            data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def minmaxscaler(data):
  min = np.amin(data)
  max = np.amax(data)
  return (data - min) / (max - min)

def data_prepare(dataset,stft_flag):
    # Vibration dataset
    TRAIN_SPLIT = int(0.8 * len(dataset))

    #last  past_history days for future_target future day
    X_train, Y_train = multivariate_data(dataset, dataset[:,0], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True,
                                                       stft_flag=stft_flag)
    X_test, Y_test = multivariate_data(dataset, dataset[:,0],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=True,
                                                   stft_flag=stft_flag)
    return X_train, Y_train , X_test, Y_test

#creating a dataset builder i.e dataloaders
class noisedDataset():
    def __init__(self, datasetnoised, datasetclean, labels, transform):
        self.noise = datasetnoised
        self.clean = datasetclean
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.noise)

    def __getitem__(self, item):
        xNoise = self.noise[item]
        xClean = self.clean[item]
        y=self.labels[item]

        if self.transform != None:  # just for using the totensor transform
            xNoise = self.transform(xNoise)
            xClean = self.transform(xClean)

        return (xNoise, xClean, y)

def StftEmbedding(dataset):
    f, t, stftvib = signal.stft(np.array(dataset['x方向振动值']), 1, nperseg=60, noverlap=59)
    stftvib = np.abs(stftvib).T
    stftvib = stftvib[:len(dataset), :]
    stftvib = pd.DataFrame(stftvib, index=dataset.index.values,
                           columns=["stft" + str(i) for i in range(stftvib.shape[1])])
    dataset = pd.concat([dataset, stftvib], axis=1)

    return dataset