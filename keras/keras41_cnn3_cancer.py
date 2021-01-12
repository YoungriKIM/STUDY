# CNN으로 구성하시오/ 2차원을 4차원으로 늘려서 하시오.

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

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

x_train = x_train.reshape(x_train.shape[0], 15, 2 ,1)
x_val = x_val.reshape(x_val.shape[0], 15, 2 ,1)
x_test = x_test.reshape(x_test.shape[0], 15, 2 ,1)

# print(x_train.shape)            #(364, 15, 2, 1)
# print(x_val.shape)              #(91, 15, 2, 1)
# print(x_test.shape)             #(114, 15, 2, 1)

#모델 구성
input1 = Input(shape=(15, 2, 1))
conv1 = Conv2D(filters=120, kernel_size=(3,2), strides=1, padding='same')(input1)
drop1 = Dropout(0.2)(conv1)
conv1 = Conv2D(90, (3,2))(drop1)
flat1 = Flatten()(conv1)
dense1 = Dense(60)(flat1)
dense1 = Dense(60)(dense1)
dense1 = Dense(60)(dense1)
output1 = Dense(2, activation='sigmoid')(dense1)
model = Model(inputs = input1, outputs = output1)

#####
# CNN 모델을 쓸 때는 분류모델 레이어의 마지막에 activation=sigmoid를 안 써도 되는 건가?
#####

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=5, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_data=(x_val, y_val), verbose=2, callbacks=[stop])

#검증, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print('loss: ', loss)

y_predict = model.predict(x_test[-5:-1])

print('y_predict_argmax: ', y_predict.argmax(axis=1))
print('y_test[-5:-1]: ',y_test[-5:-1].argmax(axis=1))

# 38-3 드랍아웃 두 번 적용 (더 좋아짐)
# loss:  [0.117403045296669, 0.9736841917037964]

# 41-3 cancer CNN       loss가 더 커짐
# loss:  [0.3340015709400177, 0.9736841917037964]
# y_predict_argmax:  [0 1 0 1]
# y_test[-5:-1]:  [0 1 0 1]