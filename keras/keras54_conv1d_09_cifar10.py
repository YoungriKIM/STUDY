# Conv1D로 바꾸시오 그리고 비교하시오

import numpy as np
from tensorflow.keras.datasets import cifar10   #10가지로 분류하는것. cifar100은 100개로 분류

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2]*3).astype('float32')/255.
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2]*3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2]*3).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv1D(filters=96, kernel_size=2, strides=1, padding='same', input_shape=(32,96), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(96, 2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(18))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='acc', patience=16, mode='max')

model.fit(x_train, y_train, epochs=300, batch_size=64, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)
print('loss:' ,loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))



# 43-2 cifar CNN
# loss: [0.040339864790439606, 0.7218999862670898]
# y_pred:  [5 8 0 0 6 6 1 6 3 1]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 54-9 conv1d
# loss: [1.1400678157806396, 0.609499990940094]