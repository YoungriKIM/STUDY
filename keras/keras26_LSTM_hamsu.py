# 23-3 을 함수형으로
# 결과치 비교

#실습 LSTM을 이용하여 프렉딕트 결과를 80으로 만드시오
import numpy as np

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,0], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape, y.shape) #(13, 3) (13,)
x = x.reshape(13,3,1)
x_pred = x_pred.reshape(1, 3, 1)
print(x.shape, y.shape) #(13, 3, 1) (13,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# model = Sequential()
# model.add(LSTM(30, activation='relu', input_shape=(3,1)))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(87, activation='relu'))
# model.add(Dense(87, activation='relu'))
# model.add(Dense(90, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(3,1))
dense1 = LSTM(30, activation='relu')(input1)
dense1 = Dense(60,activation='relu')(dense1)
dense1 = Dense(87,activation='relu')(dense1)
dense1 = Dense(87,activation='relu')(dense1)
dense1 = Dense(90,activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs =input1, outputs = output1)

#컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor = 'loss', patience=20, mode='min')
model.fit(x_train, y_train, epochs=1000, batch_size=3, validation_split=0.2, verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=3)
print('loss: ', loss)

y_pred = model.predict(x_pred)
print('y_pred: ', y_pred)

# 23-3 Seqeuntial
# loss:  [0.82417231798172]
# y_pred:  [[80.21291]]

# 26-1 functional
# loss:  [1.1014704704284668]
# y_pred:  [[85.71857]]
