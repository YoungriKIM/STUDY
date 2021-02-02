# baseline_hacking_1 을 튜닝함 > 하는 중

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization

# 데이터 맨 처음 지정 ===================
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')

# trian 데이터 지정 =======================
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values

y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))
for i , digit in enumerate(y):
    y_train[i, digit] = 1


'''
x_train = x_train.reshape(-1, 28, 28 , 1)
# print(x_train.shape)    #(2048, 28, 28, 1)
x_train = x_train/255
'''