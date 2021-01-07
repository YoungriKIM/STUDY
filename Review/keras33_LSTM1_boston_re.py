#사이킷런 데이터셋 이용 / LSTM 으로 모델링 

import numpy as np

#1. 데이터 제공
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

# print(dataset.DESCR)
# print(dataset.featrue_names)

# x는 506열에 13행
# :Number of Instances: 506
# :Number of Attributes: 13 numeric/categorical predictive.

# print(x.shape) #(506, 13)
# print(y.shape) #(506,)

print(x) #전처리 안됐음 확인
print(y) #회기 모델로 구성

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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

print(x_train.shape) #(323, 13, 1)
print(x_val.shape) #(81, 13, 1)
print(x_test.shape) #(102, 13, 1)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(32, input_shape=(13, 1), activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')

model.fit(x_train, y_train, epochs=1000, batch_size=52, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=52)
print('loss: ', loss)

y_pred = model.predict(x_test[:2])
print('y_pred: ', y_pred)
print('y_test[:2]', y_test[:2])

# loss:  [23.212236404418945, 3.754948139190674]
# y_pred:
# [[25.16304 ]
#  [19.361034]]
# y_test[:2] [19.1 17.2]