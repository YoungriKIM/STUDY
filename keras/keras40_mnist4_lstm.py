# (N, 28, 28)     CNN
# (N, 764)        Dense
# (N, 764, 1)      LSTM     > input_shape = (28*28, 1) > (28*14, 2) > (28*7, 4) 등이 더 빠를 것이다.
# lstm으로 구성

import numpy as np

#1. 데이터 불러오기
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 49, 16)
x_test = x_test.reshape(x_test.shape[0], 49, 16)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(x_train.shape, x_test.shape)   #(60000, 49, 16) (10000, 49, 16)
# print(y_train.shape, y_test.shape)   #(60000, 10) (10000, 10)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(49, 16))
lstm1 = LSTM(16, activation='relu')(input1)
dense1 = Dense(12)(lstm1)
drop1 = Dropout(0.2)(dense1)
dense1 = Dense(12)(drop1)
dense1 = Dense(12)(dense1)
output1 = Dense(10, activation='softmax')(dense1)
model = Model(inputs = input1, outputs = output1)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=16, mode='min')

model.fit(x_train, y_train, epochs=1000, batch_size=61, validation_split=0.2, verbose=1, callbacks=[stop])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=61)
print('loss: ', loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

# 40-2 mnist CNN
# loss, acc:  0.009633197449147701 0.9853999743461609     137
#
# 40-3 mnist DNN       
# loss:  0.003270471468567848
# y_pred:  [7 2 1 0 4 1 4 9 6 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]

# 40-4 mnist LSTM
# 248
# loss:  0.010707256384193897
# y_pred:  [7 2 1 0 4 1 4 9 1 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]