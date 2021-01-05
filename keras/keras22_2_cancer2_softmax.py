# 실습: 21-1 파일을 다중분류로 코딩하시오. (이중분류도 다중분류니까)

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#1. 데이터 불러오기
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x= dataset.data
y = dataset.target

#전처리
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

print(x.shape) #(569, 30)
print(y.shape) #(569, 2)

#2. 모델 구성
input1 = Input(shape=(30,))
dense1 = Dense(30, activation='relu')(input1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
dense1 = Dense(90)(dense1)
output1 = Dense(2, activation='softmax')(dense1)
model = Model(inputs = input1, outputs = output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='loss', patience=20, mode='auto')

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_data=(x_val, y_val), verbose=2, callbacks=[earlystopping])

#.평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=2)
print('loss: ', loss)
# loss:  [0.16481192409992218, 0.9912280440330505, 0.00877323467284441]

y_predict = model.predict(x[:5])

print('y_predict: ', y_predict)
#  [[1. 0.]
#  [1. 0.]
#  [1. 0.]
#  [1. 0.]
#  [1. 0.]]

print('y[:5]: ', y[:5])
# [[1. 0.]
#  [1. 0.]
#  [1. 0.]
#  [1. 0.]
#  [1. 0.]]
