# cnn모델 구성

import numpy as np

#1. 데이터
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape)          # (60000, 28, 28)
# print(x_test.shape)           # (10000, 28, 28)
# print(y_train.shape)          # (60000,)
# print(y_test.shape)           # (10000,)

# 전처리 // 4)다중분류 y벡터화 / 1)train,val 분리 / 2)minmaxscaler / 3)3차원 리쉐잎

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_train = x_train.reshape(x_train.shape[0], 16, 49).astype('float32')/255.
x_val = x_val.reshape(x_val.shape[0], 16, 49).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 16, 49).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# print(x_train.shape)        (48000, 16, 49)
# #이렇게 자르는 것은 비효율적이다. 오리지널 데이터의 세트가 28씩이었기 때문에 14나 28의 배수로 맞춰야 한다. ex) (60000, 14, 56)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM

model = Sequential()
model.add(LSTM(4, input_shape=(16, 49), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='acc', patience=16, mode='max')

# model.fit(x_train, y_train, epochs=600, batch_size=56, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])
model.fit(x_train, y_train, epochs=80, batch_size=48, validation_data=(x_val, y_val), verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=48)
print('loss:' ,loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

# 42-2 CNN
# 277
# loss: [0.01650778204202652, 0.9014000296592712]
# y_pred:  [9 2 1 1 0 1 4 6 5 7]
# y_test:  [9 2 1 1 6 1 4 6 5 7]

# 42-3 DNN
# 70
# loss: [0.017586635425686836, 0.8916000127792358]
# y_pred:  [9 2 1 1 6 1 4 6 5 7]
# y_test:  [9 2 1 1 6 1 4 6 5 7]

# 42-4 LSTM
# loss: [0.04728721082210541, 0.6898000240325928]
# y_pred:  [9 2 1 1 6 1 2 2 5 7]
# y_test:  [9 2 1 1 6 1 4 6 5 7]