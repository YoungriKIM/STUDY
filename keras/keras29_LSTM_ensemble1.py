#loss가 지표
#LSTM으로 앙상블 모델 만드시오

import numpy as np
from numpy import array

#1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,0], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], [50,60,70], [60,70,80], [70,80,90], [80,90,100], [90,100,110], [100,110,120], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])

print(x1.shape) #(13,3)
print(x2.shape) #(13,3)
print(y.shape) #(13,)
print(x1_predict.shape) #(3,)
print(x2_predict.shape) #(3,)

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) #(13, 3, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) #(13, 3, 1)
x1_predict = x1_predict.reshape(1, 3, 1) #(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1) #(1, 3, 1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test, = train_test_split(x1, x2, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, train_size=0.8, shuffle=True, random_state=311)

#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate

input1 = Input(shape=(3,1))
lstm1 = LSTM(27, activation='relu')(input1)
dense1 = Dense(18, activation='relu')(lstm1)
dense1 = Dense(9, activation='relu')(dense1)

input2 = Input(shape=(3,1))
lstm2 = LSTM(27, activation='relu')(input1)
dense2 = Dense(18, activation='relu')(lstm1)
dense2 = Dense(9, activation='relu')(dense1)

merge1 = concatenate([dense1, dense2])
middle = Dense(18)(merge1)
middle = Dense(18)(middle)

output1 = Dense(36)(middle)
output1 = Dense(18)(middle)
output1 = Dense(9)(output1)
output1 = Dense(1)(output1)

model = Model(inputs = [input1, input2], outputs = output1)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')

model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=3, validation_data=([x1_val, x2_val], y_val), verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test, batch_size=3)
print('loss: ', loss)

y_pred= model.predict([x1_predict, x2_predict])
print('y_pred: ', y_pred)

#내가 원하는 값은 85
# 436/1000
# loss:  0.05432714894413948
# y_pred:  [[85.81537]]