# 스스로 할 수 있는거 동원해서 모델 구성 conv2d 와 dnn

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 데이터 맨 처음 지정 ===================
train = pd.read_csv('../data/csv/dacon3/train.csv')
test = pd.read_csv('../data/csv/dacon3/test.csv')
sub = pd.read_csv('../data/csv/dacon3/submission.csv')

x_train_origin = train.drop(['id', 'digit', 'letter'], axis=1).values
y_train_origin = train['digit'].values
all_test = test.drop(['id', 'letter'], axis=1).values

# print(x_train_origin.shape)
# print(y_train_origin.shape)
# print(all_test_origin.shape)
# (2048, 784)
# (2048,)
# (20480, 784)

# 전처리
x_train, x_val, y_train, y_val = train_test_split(x_train_origin, y_train_origin, test_size=0.4, shuffle=True, random_state=311)

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# print(x_train.shape)
# print(x_val.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(all_test.shape)
# (1638, 784)
# (410, 784)
# (1638, 10)
# (410, 10)
# (20480, 784)

#------- for conv2d
x_train = x_train.reshape(-1, 28, 28, 1)/255
x_val = x_val.reshape(-1, 28, 28, 1)/255
all_test = all_test.reshape(-1, 28, 28, 1)/255
#------- for dnn
# x_train = x_train/255
# x_val = x_val/255
# all_test = all_test/255
# -------

# ====================================================================
# 모델 구성 --- Conv2D ---
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=4, strides=1, padding='same', input_shape=(x_train.shape[1:])))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, 4, padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, 2, padding='same'))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))

#  모델 구성 --- DNN ---
# model = Sequential()
# model.add(Dense(200, input_shape=(784, ), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(160, activation='relu'))
# model.add(Dense(80))
# model.add(Dense(80))
# model.add(Dense(40))
# model.add(Dense(10, activation = 'softmax'))

# ====================================================================

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

patience = 32
redu_lr = ReduceLROnPlateau(patience= patience, verbose=1, factor=0.5)
stop = EarlyStopping(monitor='val_acc', patience=patience*2, verbose=1, mode='acc')
mc = ModelCheckpoint(filepath= '../data/modelcheckpoint/dacon3/self_0204_1_{val_acc:.4f}.hdf5', save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=2000, batch_size=2, validation_data=(x_val, y_val), verbose=1, callbacks=[stop, redu_lr, mc])

# ====================================================================
# 예측
# model = load_model('../data/modelcheckpoint/dacon3/??')

result = model.predict(all_test, batch_size=2)

# 제출용 저장
sub['digit'] = result.argmax(1)
print(sub.head())

sub.to_csv('../data/csv/dacon3/self_0204_1.csv', index = False)
print('======save complete=====')

# =============================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')
# =============================================
# cond2d : ing