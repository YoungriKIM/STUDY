# Conv1D로 바꾸시오 그리고 비교하시오

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten, Dropout

#1. 데이터 주고
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

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

#모델 구성
input1 = Input(shape=(30,1))
conv1 = Conv1D(260, 1, activation='relu')(input1)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv1 = Conv1D(130, 1)(pool1)
pool1 = MaxPooling1D(pool_size=2)(conv1)
flat1 = Flatten()(pool1)
dense1 = Dense(30)(flat1)
output1 = Dense(2, activation='sigmoid')(dense1)

model = Model(inputs = input1, outputs = output1)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#검증, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])

print('y_predict: ', y_predict.argmax(axis=1))
print('y_test[-5:-1]: ',y_test[-5:-1].argmax(axis=1))



# 38-3
# loss:  [0.117403045296669, 0.9736841917037964]

# 54-1 conv1d
# loss:  [0.26300695538520813, 0.9473684430122375]