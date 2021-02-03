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
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')

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

# 리쉐잎 + scaler
x_train = x_train.reshape(-1, 28, 28 , 1)/255

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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_accuracy', patience=64, mode='max')
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', patience=32, factor=0.5, verbose=1)
mc = ModelCheckpoint(filepath='../data/modelcheckpoint/dacon3/baseline_0202_1.hdf5',\
    monitor='val_accuracy', save_best_only=True, mode='auto')

model.fit(x_train, y_train, batch_size=32, epochs=2000, verbose=1, validation_split=0.2, callbacks=[stop, mc, reduce_lr])

# 평가ㄴ 예측ㅇ + sub저장 =========================
x_test = test.drop(['id', 'letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)/255

sub = pd.read_csv('../data/csv/dacon3/submission.csv')
sub['digit'] = np.argmax(model.predict(x_test), axis = 1)
print(sub.head())

sub.to_csv('../data/csv/dacon3/baseline_0203_1.csv', index = False)

# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# =============================================

# epochs 수정하고 저장해라!
# mycom > 0.8578431373	