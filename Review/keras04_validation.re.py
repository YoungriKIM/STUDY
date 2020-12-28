'''
세팅하고
데이터주고
모델 구성하고
컴파일, 훈련
평가, 예측까지
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import array

x_train = np.array([1,2,3,4,5,6,7,8])
y_train = np.array([1,2,3,4,5,6,7,8])
# 머신이 훈련하는 데이터
x_validation = np.array([9,10,11,12,13])
y_validation = np.array([9,10,11,12,13])
# 머신이 검증하는 데이터
x_test = ([14,15,16,17,18])
y_test = ([14,15,16,17,18])
# 사람이 검증하는 데이터

model = Sequential()
model.add(Dense(10, input_dim=1, activation = 'linear'))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_validation, y_validation))

result = model.evaluate(x_test, y_test, batch_size=1)
print('result: ', result)

y_predict = model.predict([19])
print('y_predict: ', y_predict)