import numpy as np
from tensorflow.keras.datasets import cifar10   #10가지로 분류하는것. cifar100은 100개로 분류

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)

#전처리 // 3) y벡터화 / 2) x minmax / 1) x traintest 분리

# print(np.min(x_train), np.max(x_train))     #0 255
# print(np.min(y_train), np.max(y_train))     #0 9

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

x_train = x_train.astype('float32')/255.
x_val = x_val.astype('float32')/255.
x_test = x_test.astype('float32')/255.

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2]*x_train.shape[3])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2]*x_val.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2]*x_test.shape[3])

print(x_train.shape)            #(40000, 32, 96)
print(x_val.shape)              #(10000, 32, 96)
print(x_test.shape)             #(40000, 32, 96)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM

model = Sequential()
model.add(LSTM(16, input_shape=(32, 96), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(10))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='acc', patience=16, mode='max')

model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val), verbose=1, callbacks=[stop])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=32)
print('loss:' ,loss)

y_pred = model.predict(x_test[:10])
print('y_pred: ', y_pred.argmax(axis=1))
print('y_test: ', y_test[:10].argmax(axis=1))

# 43-2 cifar CNN
# loss: [0.040339864790439606, 0.7218999862670898]
# y_pred:  [5 8 0 0 6 6 1 6 3 1]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 43-3 cifar DNN
# loss: [0.07754093408584595, 0.3504999876022339]
# y_pred:  [5 8 8 0 4 6 5 6 5 9]
# y_test:  [3 8 8 0 6 6 1 6 3 1]

# 42-4 cifar LSTM
# loss: [0.06874780356884003, 0.4440999925136566]
# y_pred:  [2 8 8 9 4 6 9 6 2 9]
# y_test:  [3 8 8 0 6 6 1 6 3 1]