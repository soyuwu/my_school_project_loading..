# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:06:51 2023

@author: 楠木
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split


def dataset(file):
    data = pd.read_csv(file, encoding='gbk')
    # 创建LabelEncoder对象
    label_encoder = LabelEncoder()
    # 遍历每一列，将非数值型数据转换为数值型数据
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = label_encoder.fit_transform(data[column])
    # 打印转换后的数据
    label = data.iloc[:, -1:]
    dataset = data.iloc[:, :-1]
    # data1,t=ExNN_SMOTE(data)
    # print(data1)
    y = label.values
    x = dataset.values
    y = y.flatten()
    sum0 = 0
    sum1 = 0
    charge = 0
    for i in y:
        if i == 0:
            sum0 = 1 + sum0
            pass
        else:
            sum1 = sum1 + 1
    if sum1 > sum0:
        charge = 1

    scaler = MinMaxScaler()
    scaleRd = scaler.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(scaleRd, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test, abs(sum1 - sum0), charge, len(x[0])


