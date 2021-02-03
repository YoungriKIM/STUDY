# baseline_hacking_1 을 튜닝함 > 하는 중

# from google.colab import drive
# drive.mount('/content/drive')

# train, val 나누기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split

# 데이터 맨 처음 지정 ===================
train = pd.read_csv('/content/drive/MyDrive/colab_data/dacon3/train.csv')
test = pd.read_csv('/content/drive/MyDrive/colab_data/dacon3/test.csv')

# trian 데이터 지정 =======================
# x_train 지정
x_train = train.drop(['id', 'digit', 'letter'], axis=1).values

# y_train 지정 + 벡터화
y = train['digit']
y_train = np.zeros((len(y), len(y.unique())))
for i , digit in enumerate(y):
    y_train[i, digit] = 1

# print(x_train.shape)  #(2048, 784)
# print(y_train.shape)  #(2048, 10)

# train 데이터 split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle =True, random_state=311)

# print(x_train.shape)
# print(x_val.shape)
# print(y_train.shape)
# print(y_val.shape)
# (1638, 784)
# (410, 784)
# (1638, 10)
# (410, 10)

# 리쉐잎 + scaler
x_train = x_train.reshape(-1, 28, 28 , 1)/255
x_val = x_val.reshape(-1, 28, 28 , 1)/255

# 모델 구성 =========================
def mymodel(x_train):
    model = Sequential()
    model.add(Conv2D(128, kernel_size=2, input_shape=(x_train.shape[1:]), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 2, padding='same', activation='relu'))
    model.add(MaxPool2D(2,2))

    model.add(Conv2D(256, 2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, 2, padding='same', activation='relu'))
    model.add(MaxPool2D(2,2))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    return model

# 컴파일, 훈련 =========================
model = mymodel(x_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2000, verbose=1, validation_data=(x_val, y_val))

# 평가ㄴ 예측ㅇ + sub저장 =========================
x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)/255

sub = pd.read_csv('/content/drive/MyDrive/colab_data/dacon3/submission.csv')
sub['digit'] = np.argmax(model.predict(x_test), axis = 1)
print(sub.head())

sub.to_csv('/content/drive/MyDrive/colab_data/dacon3/baseline_0203_1', index = False)

# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# =============================================

# epochs 수정하고 저장해라!
# colab ing