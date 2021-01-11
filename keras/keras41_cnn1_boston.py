# CNN으로 구성하시오/ 2차원을 4차원으로 늘려서 하시오.

import numpy as np

#1. 데이터
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

# print(x.shape, y.shape)     #(506, 13) (506,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# print(x_train.shape)        #(323, 13)
# print(x_val.shape)          #(81, 13)
# print(x_test.shape)         #(102, 13)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1 ,1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1],1 ,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1 ,1)

# print(x_train.shape)        #(323, 13, 1, 1)
# print(x_val.shape)          #(81, 13, 1, 1)
# print(x_test.shape)         #(102, 13, 1, 1)

#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input

input1 = Input(shape=(13,1,1))
conv1 = Conv2D(filters=78, kernel_size=(2,1), strides=1, padding='same', activation='relu')(input1)
drop1 = Dropout(0.2)(conv1)
conv1 = Conv2D(39, (2,1))(drop1)
flat1 = Flatten()(conv1)
dense1 = Dense(39)(flat1)
dense1 = Dense(13)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')

model.fit(x_train, y_train, epochs=1000, batch_size=13, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=13)
print('loss: ', loss)

y_pred = model.predict(x_test[:5])
print('y_pred: \n', y_pred)
print('y_test: ', y_test[:5])

# 18-earlystopping 
# loss, mae :  14.958937644958496 2.9046790599823
# RMSE:  3.86767836457503
# R2:  0.821028831848539

# 41-1 boston CNN       많이 안 좋아짐
# loss:  22.685556411743164
# y_pred: 
#  [[24.685066]
#  [14.187754]
#  [20.230017]
#  [26.658447]
#  [18.59134 ]]
# y_test:  [19.1 17.2 17.8 23.4 14.6]