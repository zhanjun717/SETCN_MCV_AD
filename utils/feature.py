#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import time
import pandas as pd
import numpy as np

def Get_Statistics(df , T):
    print('# Get_Statistics...')
    time_start = time.time()

    df_mean = df.rolling(T).mean()      # 均值
    df_kurt = df.rolling(T).kurt()      # 峭度
    df_var = df.rolling(T).var()        # 方差
    # df_cov = df.rolling(T).cov()        # 协方差
    # df_rms = df.rolling(T).apply(rms)   # 均方根值
    df_peak = df.rolling(T).max() - df.rolling(T).min()     # 峰峰值

    time_end = time.time()
    print('totally cost ： {} s'.format(time_end - time_start))

    return df_mean,df_kurt,df_var,df_peak

def fearture(dataset):
    # 特征工程
    print('# Feature Engineering...')
    T = 60                                                              # 统计滑窗宽度
    df_mean, df_kurt, df_var, df_peak = Get_Statistics(dataset , T)     # 求取统计值
    df_mean.dropna(axis=0, how='any', inplace=True)
    df_mean.reset_index(inplace=True)

    df_kurt.dropna(axis=0, how='any', inplace=True)
    df_kurt.reset_index(inplace=True)

    df_var.dropna(axis=0, how='any', inplace=True)
    df_var.reset_index(inplace=True)

    df_peak.dropna(axis=0, how='any', inplace=True)
    df_peak.reset_index(inplace=True)

    # 10min change rate
    cr_time = T
    df_mean_10min = dataset.rolling(cr_time).mean()                          # 10min风速变化率
    df_mean_cr = abs(df_mean_10min.diff() / (cr_time)) * 100                 # 轮毂转速变化率 Change Rate
    df_mean_cr.dropna(axis=0, how='any', inplace=True)
    df_mean_cr.reset_index(inplace=True)

    cr_time = T*10
    df_std_10min = dataset.rolling(cr_time).std()                              # 10min湍流强度
    turb_10min = pd.DataFrame(df_std_10min['机舱气象站风速'] / df_mean_10min['机舱气象站风速'])
    turb_10min.dropna(axis=0, how='any', inplace=True)
    turb_10min.reset_index(inplace=True)
    print(turb_10min)

    # 1hz
    df_new = pd.DataFrame()                                             # 构建新特征
    df_new[['机舱气象站风速', 'x方向振动值', '轮毂转速']] = dataset[['机舱气象站风速', 'x方向振动值', '轮毂转速']]
    df_new[['风速CR', 'x_CR', '转速CR']] = df_mean_cr[['机舱气象站风速', 'x方向振动值', '轮毂转速']]
    # df_new['湍流强度'] = turb_10min['机舱气象站风速']
    df_new['轮毂转速均值'] = df_mean['轮毂转速']
    df_new['x振动均值'] = df_mean['x方向振动值']
    df_new['x振动峭度'] = df_kurt['x方向振动值']
    df_new['x振动方差'] = df_var['x方向振动值']
    df_new['x振动峰峰值'] = df_peak['x方向振动值']

    # # 0.1hz
    # df_new = pd.DataFrame()                                             # 构建新特征
    # df_new[['风速CR', 'x_CR', '转速CR']] = df_mean_cr[['机舱气象站风速', 'x方向振动值', '轮毂转速']]
    # df_new['湍流强度'] = turb_10min['机舱气象站风速']
    # df_new[['机舱气象站风速平均','轮毂转速均值']] = df_mean[['机舱气象站风速','轮毂转速']]
    # df_new['x振动均值'] = df_mean['x方向振动值']
    # df_new['x振动峭度'] = df_kurt['x方向振动值']
    # df_new['x振动方差'] = df_var['x方向振动值']
    # df_new['x振动峰峰值'] = df_peak['x方向振动值']

    return df_new

def preprocess(dataset):
    numb_bool = dataset['x方向振动值'].apply(lambda x: (x != -10))
    dataset = dataset[numb_bool]
    # numb_bool = dataset['x方向振动值'].apply(lambda x: (x != 0))
    # dataset = dataset[numb_bool]

    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(axis=0, how='any', inplace=True)
    dataset = dataset.reset_index(drop=True)

    return dataset

def fearture_engineering():
    ## 数据准备
    print('# Data load...')
    time_now = time.time()
    train_data = pd.read_parquet('../data/TB004_train_normal.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])
    test_data = pd.read_parquet('../data/TB004_2018_abnormal.parquet', columns=['机舱气象站风速', 'x方向振动值', '轮毂转速'])

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
    time_end = time.time()
    print("totally cost : {} s".format(time_end - time_now))

    # 特征工程
    df_new_train = fearture(train_data)
    df_new_test = fearture(test_data)

    df_new_train.dropna(axis=0, how='any', inplace=True)
    df_new_train.to_parquet("../data/TB004-train-1hz.parquet", index=False)

    df_new_test.dropna(axis=0, how='any', inplace=True)
    df_new_test.to_parquet("../data/TB004-test-1hz.parquet", index=False)

if __name__ == '__main__':
    fearture_engineering()