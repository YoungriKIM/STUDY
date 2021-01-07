# 와인 데이터셋으로 LSTM 만들기
import numpy as np

#1. 데이터
from sklearn.datasets import load_wine

dataset = load_wine()
x = dataset.data
y = dataset.target

# print(dataset.DESCR) #178행에 13열인 x 
# print(x) #전처리 안되어 있음 확인
# print(y) # 0,1,2 로 다중분류임. y 벡터화 해야 함

#   y_벡터화
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

#   train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=311)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

#   MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#   3차원으로 reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# print(x_train.shape)    #(113, 13, 1)
# print(x_val.shape)      #(29, 13, 1)
# print(x_test.shape)     #(36, 13, 1)

# print(y.shape) #(178, 3)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(60, input_shape=(13,1), activation='relu'))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='loss', patience=20, mode='min')

model.fit(x_train, y_train, epochs=1000, batch_size=20, validation_data = (x_val, y_val), verbose=2, callbacks=[stop])

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=20)
print('loss: ', loss)

y_pred = model.predict(x_test[:5])
print('y_pred[:5]: ', y_pred.argmax(axis=1))
print('y_test[:5]: ', y_test[:5].argmax(axis=1))

#################################
# loss:  [0.37645065784454346, 0.9166666865348816]
# y_pred[:5]:  [1 1 2 1 1]
# y_test[:5]:  [1 1 2 1 1]