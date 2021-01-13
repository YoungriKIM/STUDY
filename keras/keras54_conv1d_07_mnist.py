# Conv1D로 바꾸시오 그리고 비교하시오

import numpy as np

#1. 데이터 불러오기
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000,)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, MaxPooling1D, Dropout

model = Sequential()
model.add(Conv1D(200, 2, input_shape=(28, 28), activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Dropout(0.2))
model.add(Conv1D(100, 1, input_shape=(3,1), activation='relu'))
model.add(Conv1D(80, 1, input_shape=(3,1), activation='relu'))
model.add(Flatten())
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_acc', patience=16, mode='max')

model.fit(x_train, y_train, epochs=10, batch_size=69, validation_split=0.2, verbose=1, callbacks=[stop])

#. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=69)
print('loss: ', loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))



# 40-2 mnist CNN
# loss, acc:  0.009633197449147701 0.9853999743461609     137

# 40-3 mnist DNN       
# loss:  [0.10886456072330475, 0.9815000295639038]
# y_pred:  [7 2 1 0 4 1 4 9 6 9]
# y_test:  [7 2 1 0 4 1 4 9 5 9]

# 54-7 conv1d
# loss:  [0.06335253268480301, 0.9793000221252441]