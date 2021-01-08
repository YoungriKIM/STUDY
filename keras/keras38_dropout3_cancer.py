#실습 드랍아웃 적용

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

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

#모델 구성
input1 = Input(shape=(30,))
dense1 = Dense(120, activation='relu')(input1)
dropout1 = Dropout(0.2)(dense1) # 이번에 붙인 부분
dense1 = Dense(120)(dropout1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(60)(dropout1)
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

print('y_predict: ', y_predict)
print('y_predict_argmax: ', y_predict.argmax(axis=1)) #0이 열, 1이 행

print('y_test[-5:-1]: ',y_test[-5:-1])


# print(y_predict.argmax(axis=1).shape) #(4,)
# print(y_test[-5:-1].shape) #(4,2)

# 21-2 드랍아웃 미적용
# loss:  [0.31692570447921753, 0.9561403393745422]

# 38-3 드랍아웃 한 번 적용(loss가 줄어서 더 좋아짐)
# loss:  [0.15717896819114685, 0.9561403393745422]

# 38-3 드랍아웃 두 번 적용 (더 좋아짐)
# loss:  [0.117403045296669, 0.9736841917037964]