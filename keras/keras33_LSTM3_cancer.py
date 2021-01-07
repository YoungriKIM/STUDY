#사이킷런 데이터셋 이용 / LSTM 으로 모델링 / Dense와 성능 비교 / 이진분류

# 21-1의 predict 값이 소수점이 아니라 0.1 로 나오게 할 것

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM

#1. 데이터 주고
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

#전처리(y벡터화, 트레인테스트나누기, 민맥스스케일러)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=33)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=33)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(364, 30, 1)
print(x_val.shape) #(91, 30, 1)
print(x_test.shape) #(114, 30, 1)

print(y.shape) #(569,2)

#모델 구성
input1 = Input(shape=(30,1))
dense1 = LSTM(120, activation='relu')(input1)
dense1 = Dense(120)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
output1 = Dense(2, activation='sigmoid')(dense1)
model = Model(inputs = input1, outputs = output1)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=5, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#검증, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])

print('y_predict_argmax: ', y_predict.argmax(axis=1)) #0이 열, 1이 행

# 비교용 21-1 Dense
# 21
# loss:  [0.12744367122650146, 0.9561403393745422]
# [0 1 0 1]

# # 33-3 LSTM
# loss:  [0.20369993150234222, 0.9649122953414917]
# y_predict_argmax:  [0 1 0 1]