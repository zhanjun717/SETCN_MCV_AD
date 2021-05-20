#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''

import matplotlib.pyplot as plt
import torch
from utils.biuld_dataset import *
import pandas as pd
from torch.utils.data import DataLoader,Dataset,TensorDataset
from sklearn import preprocessing
# from train_main import predict
import time
from scipy import signal

from scipy.fftpack import fft,ifft

batch_size = 64
label_transform = False


def mean5_3(Series):
    n=len(Series)
    a=Series
    b=Series.copy()
    for i in range(3):
        b[0] = (69*a[0] + 4*(a[1] + a[3]) - 6*a[2] - a[4]) /70
        b[1] = (2*(a[0] + a[4]) +27*a[1] + 12*a[2] - 8*a[3]) /35
        for j in range(2,n-2):
            b[j] = (-3*(a[j-2] + a[j+2]) + 12*(a[j-1] + a[j+1]) + 17*a[j]) /35
        b[n-2] = (2*(a[n-1] + a[n-5]) + 27*a[n-2] + 12*a[n-3] - 8*a[n-4]) /35
        b[n-1] = (69*a[n-1] + 4*(a[n-2] + a[n-4]) - 6*a[n-3] -a[n-5]) /70
        a=b.copy()
    return a

def data_visu():
    # dataset013_l = pd.read_parquet('./data/TB013_2015.parquet')

    # dataset013_2015 = pd.read_parquet('./data/TB013_2015.parquet', columns=['记录时间', '机舱气象站风速', 'x方向振动值', 'y方向振动值'])
    # dataset013_2016 = pd.read_parquet('./data/TB013_2016.parquet', columns=['记录时间', '机舱气象站风速', 'x方向振动值', 'y方向振动值'])
    #
    # dataset013 = pd.concat([dataset013_2015, dataset013_2016], axis=0).reset_index(drop=True)
    # dataset013["记录时间"] = pd.to_datetime(dataset013["记录时间"], format='%Y-%m-%d %H:%M:%S')
    # dataset013.index = dataset013["记录时间"].values
    #
    # # dataset = pd.read_parquet('./data/TB012_2018.parquet', columns=['机舱气象站风速', 'x方向振动值', 'y方向振动值'])
    # normal_dataset = dataset013['2016-01-29':'2016-02-08']
    # abnormal_dataset = dataset013['2016-02-22':'2016-02-22']
    #
    # dataset = dataset013['2016-02-14 00:00:00':'2016-02-14 00:10:00']
    # dataset_mean = dataset.rolling(10).mean()

    # plt.figure(0)
    # plt.subplot(211)
    # plt.plot(dataset['机舱气象站风速'])
    # plt.plot(dataset_mean['机舱气象站风速'], color="yellow")
    #
    # plt.subplot(212)
    # plt.plot(dataset['x方向振动值'])
    # plt.plot(dataset_mean['x方向振动值'], color="yellow")
    # # plt.plot(dataset['x方向振动值'].rolling(60).apply(mean5_3(dataset['x方向振动值'].values)), color="red")
    #
    # # plt.subplot(313)
    # # plt.plot(dataset['y方向振动值'])
    # # plt.plot(dataset['y方向振动值'].rolling(60).mean(), color="yellow")
    #
    # plt.show()
    dataset = pd.read_parquet('./data/TB013_2016.parquet', columns=['记录时间', '机舱气象站风速', 'x方向振动值', 'y方向振动值'])
    dataset["记录时间"] = pd.to_datetime(dataset["记录时间"], format='%Y-%m-%d %H:%M:%S')
    dataset.index = dataset["记录时间"].values

    dataset = dataset['2016-02-14 00:00:00':'2016-02-14 00:02:00']

    fvibx = fft(np.array(dataset['x方向振动值']))
    f,t,stftvib = signal.stft(np.array(dataset['x方向振动值']),1 , nperseg=60, noverlap=59)
    stftvib = np.abs(stftvib)
    normalization_vibx = np.abs(fvibx)/len(dataset)
    print(fvibx)
    # plt.figure(1)
    # plt.pcolormesh(t, f, np.abs(Zxx))
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

def data_select():

    ###################
    dataset013_2015 = pd.read_parquet('./data/TB013_2015.parquet', columns=['记录时间', 'x方向振动值', 'y方向振动值', '机舱气象站风速', '轮毂转速', '变桨电机1电流', '变桨电机2电流', '变桨电机3电流', '液压制动压力', '发电机转矩', '机舱温度', '5秒偏航对风平均值', '变频器发电机侧功率'])
    dataset013_2016 = pd.read_parquet('./data/TB013_2016.parquet', columns=['记录时间', 'x方向振动值', 'y方向振动值', '机舱气象站风速', '轮毂转速', '变桨电机1电流', '变桨电机2电流', '变桨电机3电流', '液压制动压力', '发电机转矩', '机舱温度', '5秒偏航对风平均值', '变频器发电机侧功率'])

    dataset013 = pd.concat([dataset013_2015, dataset013_2016], axis=0).reset_index(drop=True)
    dataset013["记录时间"] = pd.to_datetime(dataset013["记录时间"], format='%Y-%m-%d %H:%M:%S')
    dataset013.index = dataset013["记录时间"].values

    normal_dataset = dataset013['2016-01-29':'2016-02-08']
    abnormal_dataset = dataset013['2016-02-22':'2016-02-22']

    pd.DataFrame(normal_dataset).to_parquet('./data/TB013_train_normal.parquet')
    pd.DataFrame(abnormal_dataset).to_parquet('./data/TB013_test_abnormal.parquet')

    #################
    # dataset_2017 = pd.read_parquet('../data/TB020_2017.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])
    # dataset20_2018 = pd.read_parquet('../data/TB020_2018.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])
    # dataset_2019 = pd.read_parquet('../data/TB020_2019.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])

    # dataset04_2018 = pd.read_parquet('../data/TB004_2018.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])
    # dataset04_2017 = pd.read_parquet('../data/TB004_2017.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])
    # dataset04_2016 = pd.read_parquet('../data/TB004_2016.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])

    # dataset = pd.concat([dataset04_2017, dataset04_2018], axis=0).reset_index(drop=True)
    # dataset = dataset04_2018.loc[0:1000000,:]
    # pd.DataFrame(dataset).to_parquet('./data/TB004_train_normal.parquet')
    # #
    # # print(dataset20_2018.describe())
    # print(dataset.describe())

    # plt.figure(0)
    # plt.subplot(211)
    # plt.plot(dataset['机舱气象站风速'])
    # plt.plot(dataset['机舱气象站风速'].rolling(60).mean(), color="yellow")
    #
    # plt.subplot(212)
    # plt.plot(dataset['x方向振动值'])
    # plt.plot(dataset['x方向振动值'].rolling(60).mean(), color="yellow")

    # plt.subplot(313)
    # plt.plot(dataset['轮毂转速'])
    # plt.plot(dataset['轮毂转速'].rolling(60).mean(), color="yellow")

    # plt.show()

    # test_dataset = dataset04_2018.loc[1000000:1400000,:]
    # numb_bool = test_dataset['x方向振动值'].apply(lambda x: (x != -10))
    # test_dataset = test_dataset[numb_bool]
    # numb_bool = test_dataset['x方向振动值'].apply(lambda x: (x != 0))
    # test_dataset = test_dataset[numb_bool]
    #
    # test_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    # test_dataset.dropna(axis=0, how='any', inplace=True)
    # test_dataset = test_dataset.reset_index(drop=True)
    #
    # pd.DataFrame(test_dataset).to_parquet('./data/TB004_2018_abnormal.parquet')
    #
    #
    # print(test_dataset.shape)
    #
    # plt.figure(0)
    # plt.plot(test_dataset['x方向振动值'])
    # plt.show()

def data_predict():
    # 读取数据
    # data_select()
    # freq = '1hz'
    # dataset = pd.read_parquet("./data/HD04/TB004_2018_abnormal.parquet",
    #                           columns=['记录时间', '机舱气象站风速', 'x方向振动值', '轮毂转速', '叶片1角度', '5秒偏航对风平均值', '变频器发电机侧功率'])
    #
    # y_label = dataset['x方向振动值']
    # dataset['x_t-1方向振动值'] = dataset['x方向振动值'].shift(axis=0)
    # dataset.drop(['x方向振动值'], axis=1, inplace=True)
    #
    # df_new = pd.DataFrame()  # 构建新特征
    # df_new['label'] = y_label
    # df_new[['x_t-1方向振动值', '机舱气象站风速', '轮毂转速', '叶片1角度', '5秒偏航对风平均值', '变频器发电机侧功率']] = \
    #     dataset[['x_t-1方向振动值', '机舱气象站风速', '轮毂转速', '叶片1角度', '5秒偏航对风平均值', '变频器发电机侧功率']]
    #
    # df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df_new.dropna(axis=0, how='any', inplace=True)
    # dataset = df_new.reset_index(drop=True)

    dataset = pd.read_parquet("./data/HD04/TB004_2018_abnormal.parquet")

    data_1hz = np.array(dataset.drop(['label'], axis=1))
    lable_1hz = np.array(dataset["label"]).reshape(-1, 1)

    dataset_train = pd.read_parquet('./data/HD04/TB004_train_normal_fornormalizer.parquet')
    dataset_normalizer = preprocessing.StandardScaler().fit(dataset_train)
    data_normal_1hz = dataset_normalizer.transform(data_1hz)

    lable_train = pd.read_parquet('./data/HD04/TB004_train_label_fornormalizer.parquet')
    y_train_normalizer = preprocessing.StandardScaler().fit(lable_train)
    lable_normal_1hz = y_train_normalizer.transform(lable_1hz)

    if label_transform:
        dataset = np.hstack((lable_normal_1hz, data_normal_1hz))
    else:
        dataset = np.hstack((lable_1hz, data_normal_1hz))

    X_test, Y_test = multivariate_data(dataset, dataset[:, 0], 0,
                                         int(len(dataset)), past_history,
                                         future_target, STEP,
                                         single_step=True)

    # creating the dataset
    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(Y_test, dtype=torch.float))

    # creating the dataloader
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 加载训练好的模型
    model = torch.load('./models/checkpoint_best.pth')

    # 预测
    mdhms = time.strftime('%d%H%M', time.localtime(time.time()))
    model.eval()
    predict_loss, total_loss, pridict_output, target_input = predict(test_loader, model)

    plt.figure(0)
    plt.title('Test loss')
    plt.plot(total_loss, color='blue', label='loss', linewidth=2)
    plt.savefig('./figure' + '/total_loss_' + mdhms + '.png')
    plt.show()

    xval_loss = torch.cat(total_loss, dim=0).cpu().detach().numpy()
    pd.DataFrame(xval_loss, columns=['xval_loss']).to_parquet('./data/HD04/TB004-train-xloss-1hz.parquet')

    out_val = torch.cat(pridict_output,dim=0).cpu().detach().numpy()
    Y_val = torch.cat(target_input,dim=0).cpu().detach().numpy()

    if label_transform:
        out_val_raw = y_train_normalizer.inverse_transform(out_val.reshape(-1,1))
        Y_val_raw = y_train_normalizer.inverse_transform(Y_val.reshape(-1, 1))
    else:
        out_val_raw = out_val
        Y_val_raw = Y_val

    res = out_val_raw - Y_val_raw
    # result_store = pd.concat([res, X_test], axis=1, ignore_index=True)
    # pd.DataFrame(result_store).to_parquet('./data/TB004-test-result-1hz.parquet')

    plt.figure(1, figsize=(10, 6))
    plt.subplot(211)
    plt.plot(pd.DataFrame(out_val_raw), color='yellow', label='predict', linewidth=2)
    plt.plot(pd.DataFrame(Y_val_raw), color='blue', label='lable', linewidth=2)
    plt.legend()

    plt.subplot(212)
    plt.plot(res, color='blue', label='res', linewidth=2)
    plt.plot(pd.DataFrame(res).rolling(30).mean(), color='yellow', label='res_mean', linewidth=2)
    plt.legend()

    plt.savefig('./figure' + '/val_res_' + mdhms + '.png')
    plt.show()

if __name__ == '__main__':
    # data_predict()
    # data_select()
    data_visu()
